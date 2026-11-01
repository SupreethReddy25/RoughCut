"""The event pipeline (PLAN 14 Phase 3): event -> reward mapping -> policy update -> EDL rebuild -> re-score -> response.

`POST /events` is the single entry point for every user action. Each request:
  1. validates the action against the *current* edit state,
  2. writes one or more Event rows (a transcript edit can map to several; they share a `group` and undo together),
  3. applies the learnable ones to the user's policy through `Learner` (the same code path replay uses),
  4. re-scores untouched suggestions / handles nudges (new rows, undoable),
  5. returns the rebuilt edit state.
"""
from __future__ import annotations

import time
import uuid

from sqlmodel import Session

from app import music as music_lib
from app.api.io import EventIn, ManualEditIn, StateOut
from app.appctx import AppContext
from app.convert import event_from_row, suggestion_from_row
from app.models import EventRow, SuggestionRow, User
from app.policy import context
from app.policy.reward import (
    BINARY,
    DEFAULT,
    LEARNED_KINDS,
    ManualEdit,
    Mapping,
    classify_manual_edit,
    filler_word_ids,
    nudge_offsets,
    updates_for,
)
from app.schemas import TextOverlay
from app.state import (
    DEFAULT_KEYS,
    NEUTRAL,
    mapping_to_payload,
    validate_music,
    validate_setting,
    validate_speed,
)
from app.svc.bundle import (
    EXPLICIT,
    ApiError,
    Bundle,
    event_dict,
    load_bundle,
    redo_target,
    state_out,
    undo_target,
)
from app.svc.rescore import rescore

READY = ("ready", "edited", "exported")
MAX_TEXT = 120


# --------------------------------------------------------------------------- helpers
def require_ready(b: Bundle) -> None:
    if b.video.status not in READY or b.analysis is None:
        raise ApiError(409, f"project is {b.video.status}, not ready", project_status=b.video.status)


def get_suggestion(b: Bundle, sid: int | None) -> SuggestionRow:
    if sid is None:
        raise ApiError(422, "suggestion_id is required for this kind of event")
    s = b.sug_by_id.get(sid)
    if s is None:
        raise ApiError(404, f"suggestion {sid} not found in this project")
    if s.superseded:
        newer = [r.id for r in b.sug_rows if r.rescored_from == sid and not r.superseded]
        raise ApiError(409, f"suggestion {sid} was replaced by a re-score", replaced_by=newer[0] if newer else None)
    return s


def arms_of(s: SuggestionRow) -> list[str]:
    return list(s.candidates_json) or context.arms_for(s.decision_type, s.context_raw_json)


def mapping_for(b: Bundle, s: SuggestionRow, final: str) -> Mapping:
    """The decision change `final` means for suggestion `s` (what replay applies to the edit)."""
    dt = s.decision_type
    word_ids: list[int] = []
    word_id: int | None = None
    if dt == "filler":
        fu = next(u for u in b.analysis.uf.units if u.id == s.target_id)
        word_ids = sorted(filler_word_ids(b.analysis.uf, fu))
    elif dt == "emphasis":
        word_ids = [int(s.context_raw_json[f"c{k}_word_id"]) for k in range(int(s.context_raw_json["n_cand"]))]
        if final != "NONE":
            word_id = word_ids[int(final.split("_")[1])]
    return Mapping("override", dt, s.target_id, final, word_ids, word_id=word_id)


def find_suggestion(b: Bundle, dt: str, target_id: int) -> SuggestionRow | None:
    return next((s for s in b.active if s.decision_type == dt and s.target_id == target_id), None)


def _decision_action(b: Bundle, dt: str, target_id: int) -> str:
    st = b.state.decisions
    return st.zoom.get(target_id, "NONE") if dt == "zoom" else st.action(target_id)


def _event(b: Bundle, kind: str, suggestion_id: int | None, payload: dict, ups: list | None = None) -> EventRow:
    reward = ups[0].reward if ups else None
    weight = ups[0].weight if ups else None
    return EventRow(user_id=b.video.user_id, video_id=b.video.id, kind=kind, suggestion_id=suggestion_id,
                    payload_json=payload, reward=reward, weight=weight)


def _learning_updates(s: SuggestionRow, kind: str, payload: dict):
    return updates_for(kind, s.decision_type, payload.get("from_action") or s.proposed_action, arms_of(s),
                       payload.get("final_action"), DEFAULT, manual=payload.get("via") == "manual")


# --------------------------------------------------------------------------- per-kind builders (no DB writes)
def _explicit(b: Bundle, body: EventIn) -> list[EventRow]:
    s = get_suggestion(b, body.suggestion_id)
    cur = b.action_of(s)
    arms = arms_of(s)
    payload: dict = {}
    if cur != s.proposed_action:
        payload["from_action"] = cur
    fin = body.final_action
    if fin is not None and fin not in arms:
        raise ApiError(422, f"{fin!r} is not an action of this {s.decision_type} decision: {arms}")
    if fin is not None and fin == cur and body.kind != "accept":
        raise ApiError(409, f"the edit already has {fin!r}: nothing to change (use accept)")
    if body.kind == "accept":
        mine = b.explicit.get(s.id, [])
        if mine and mine[-1].kind == "accept":
            raise ApiError(409, "this suggestion is already accepted")
        ups = _learning_updates(s, "accept", payload)
    elif body.kind == "override":
        if fin is None:
            raise ApiError(422, "override needs final_action")
        payload.update(final_action=fin, via="panel", mapping=mapping_to_payload(mapping_for(b, s, fin)))
        ups = _learning_updates(s, "override", payload)
    else:  # reject
        if s.decision_type in BINARY:
            others = [a for a in arms if a != cur]
            applied = fin or others[0]
        else:
            applied = fin or NEUTRAL[s.decision_type]
        payload.update(final_action=fin, applied=applied, mapping=mapping_to_payload(mapping_for(b, s, applied)))
        ups = _learning_updates(s, "reject", payload)
    return [_event(b, body.kind, s.id, payload, ups)]


def _manual_edit(b: Bundle, e: ManualEditIn) -> ManualEdit:
    edit = ManualEdit(kind=e.kind, word_ids=list(e.word_ids), unit_id=e.unit_id, option=e.option, out_gap=e.out_gap,
                      word_id=e.word_id, line_word_ids=list(e.line_word_ids))
    if e.kind == "set_emphasis":
        edit.line_idx = -1
        line = set(e.line_word_ids)
        for s in b.active:  # the D5 decision of the caption line holding these words
            if s.decision_type == "emphasis":
                cands = [int(s.context_raw_json[f"c{k}_word_id"]) for k in range(int(s.context_raw_json["n_cand"]))]
                if line & set(cands):
                    edit.line_idx, edit.candidate_word_ids = s.target_id, cands
                    break
    return edit


def _manual(b: Bundle, body: EventIn) -> list[EventRow]:
    if body.edit is None:
        raise ApiError(422, "manual events need `edit`")
    uf = b.analysis.uf
    n_words = len(uf.words)
    bad = [w for w in [*body.edit.word_ids, *body.edit.line_word_ids] if not 0 <= w < n_words]
    if bad:
        raise ApiError(422, f"unknown word ids {bad[:5]}")
    try:
        maps = classify_manual_edit(_manual_edit(b, body.edit), uf, b.state.decisions)
    except ValueError as e:
        raise ApiError(422, str(e)) from None
    events: list[EventRow] = []
    for m in maps:
        payload = {"mapping": mapping_to_payload(m), "via": "manual", "note": m.note}
        if m.kind == "override":
            s = find_suggestion(b, m.decision_type, m.target_id) if m.target_id is not None and m.target_id >= 0 else None
            cur = (b.action_of(s) if s is not None else
                   _decision_action(b, m.decision_type, m.target_id) if m.decision_type != "emphasis" else None)
            if cur is not None and cur == m.final_action:
                continue  # already so: no-op
            if s is None and m.decision_type == "emphasis" and m.final_action == "NONE"                     and not set(m.word_ids) & b.state.decisions.emphasis:
                continue  # clearing emphasis on a line that has none
            if s is None:  # an applied but unlearnable change (e.g. a filler that had no clean gap)
                events.append(_event(b, "fine_cut", None, payload))
                continue
            if cur != s.proposed_action:
                payload["from_action"] = cur
            payload["final_action"] = m.final_action
            events.append(_event(b, "override", s.id, payload, _learning_updates(s, "override", payload)))
        elif m.kind == "fine_cut" or (m.kind == "edit_only" and m.decision_type == "emphasis"):
            events.append(_event(b, "fine_cut", None, payload))
    return events


def _nudge(b: Bundle, body: EventIn) -> list[EventRow]:
    name = str(body.payload.get("name", ""))
    try:
        magnitude = float(body.payload.get("magnitude", 0.5))
        if not 0 < magnitude <= 3:
            raise ValueError("magnitude must be in (0, 3]")
        offsets = nudge_offsets(name, magnitude)
    except (ValueError, TypeError) as e:
        raise ApiError(422, str(e)) from None
    return [_event(b, "nudge", None, {"name": name, "magnitude": magnitude, "offsets": offsets})]


def _crop(b: Bundle, body: EventIn) -> list[EventRow]:
    p = body.payload
    try:
        uid, x, y = int(p["unit_id"]), float(p["x"]), float(p["y"])
    except (KeyError, TypeError, ValueError):
        raise ApiError(422, "crop needs payload {unit_id, x, y}") from None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ApiError(422, "x and y must be in [0, 1]")
    if not any(u.id == uid and u.kind == "speech" for u in b.analysis.uf.units):
        raise ApiError(422, f"unit {uid} is not a speech unit")
    return [_event(b, "crop", None, {"unit_id": uid, "x": x, "y": y})]


def _caption_text(b: Bundle, body: EventIn) -> list[EventRow]:
    p = body.payload
    try:
        wid, text = int(p["word_id"]), str(p["text"]).strip()
    except (KeyError, TypeError, ValueError):
        raise ApiError(422, "caption_text needs payload {word_id, text}") from None
    if not 0 <= wid < len(b.analysis.uf.words):
        raise ApiError(422, f"unknown word {wid}")
    if not text or len(text) > 80:
        raise ApiError(422, "caption text must be 1-80 characters")
    return [_event(b, "caption_text", None, {"word_id": wid, "text": text})]


def _text_overlay(b: Bundle, body: EventIn) -> list[EventRow]:
    p = body.payload
    op = p.get("op")
    if op not in ("add", "update", "delete"):
        raise ApiError(422, "text_overlay needs payload.op in add|update|delete")
    if op == "delete":
        oid = str(p.get("id", ""))
        if oid not in b.state.texts:
            raise ApiError(404, f"no text overlay {oid!r}")
        return [_event(b, "text_overlay", None, {"op": "delete", "overlay": {"id": oid}})]
    if op == "update":
        base = b.state.texts.get(str(p.get("id", "")))
        if base is None:
            raise ApiError(404, f"no text overlay {p.get('id')!r}")
        data = {**base.model_dump(), **{k: v for k, v in p.items() if k in TextOverlay.model_fields and k != "id"}}
    else:
        data = {"id": uuid.uuid4().hex[:8], **{k: v for k, v in p.items() if k in TextOverlay.model_fields and k != "id"}}
    try:
        ov = TextOverlay(**data)
    except ValueError as e:
        raise ApiError(422, f"invalid text overlay: {e}") from None
    from app.render.ass import load_styles

    if not ov.text.strip() or len(ov.text) > MAX_TEXT:
        raise ApiError(422, f"overlay text must be 1-{MAX_TEXT} characters")
    if not 0 <= ov.t0 < ov.t1:
        raise ApiError(422, "overlay needs 0 <= t0 < t1")
    if ov.position == "custom" and (ov.y is None or not 0 <= ov.y <= 1):
        raise ApiError(422, "a custom overlay position needs y in [0, 1]")
    if ov.style_id not in load_styles()[1]["styles"]:
        raise ApiError(422, f"unknown text style {ov.style_id!r}")
    return [_event(b, "text_overlay", None, {"op": op, "overlay": ov.model_dump()})]


def _speed(b: Bundle, body: EventIn) -> list[EventRow]:
    try:
        v = validate_speed(body.payload.get("value"))
    except ValueError as e:
        raise ApiError(422, str(e)) from None
    return [_event(b, "speed", None, {"value": v})]


def _setting(b: Bundle, body: EventIn) -> list[EventRow]:
    try:
        key = str(body.payload.get("key", ""))
        value = validate_setting(key, body.payload.get("value"))
    except ValueError as e:
        raise ApiError(422, str(e)) from None
    return [_event(b, "setting", None, {"key": key, "value": value})]


def _music(ctx: AppContext, b: Bundle, body: EventIn) -> list[EventRow]:
    p = body.payload
    try:
        spec = validate_music(p) if p.get("track_id") else None
    except ValueError as e:
        raise ApiError(422, str(e)) from None
    if spec is not None and music_lib.track_path(ctx.data_dir, b.video.user_id, spec["track_id"]) is None:
        raise ApiError(404, f"no music track {spec['track_id']!r}")
    cur = b.state.music.model_dump() if b.state.music else None
    if spec == cur:
        return []
    return [_event(b, "music", None, {"music": spec})]


def _remember_defaults(session: Session, b: Bundle, events: list[EventRow]) -> None:
    """Settings, speed and music the user picks become the starting point of their next projects (PLAN 8.2 / 8.10)."""
    upd: dict = {}
    for e in events:
        if e.kind == "setting" and e.payload_json["key"] in DEFAULT_KEYS:
            upd[e.payload_json["key"]] = e.payload_json["value"]
        elif e.kind == "speed":
            upd["speed"] = e.payload_json["value"]
        elif e.kind == "music":
            upd["music"] = e.payload_json["music"]
    user = session.get(User, b.video.user_id) if upd else None
    if user is not None:
        user.defaults_json = {**user.defaults_json, **upd}
        session.add(user)


def _rating(b: Bundle, body: EventIn) -> list[EventRow]:
    value = body.payload.get("value")
    if value not in (-1, 0, 1):
        raise ApiError(422, "rating needs payload.value of -1, 0 or 1")
    return [_event(b, "rating", None, {"value": value})]


BUILDERS = {"rating": _rating, "accept": _explicit, "reject": _explicit, "override": _explicit, "manual": _manual, "nudge": _nudge,
            "crop": _crop, "caption_text": _caption_text, "text_overlay": _text_overlay, "speed": _speed,
            "setting": _setting}


# --------------------------------------------------------------------------- entry points
def handle_event(ctx: AppContext, session: Session, video_pk: int, body: EventIn) -> StateOut:
    with ctx.locks(("video", video_pk)):
        b = load_bundle(ctx, session, video_pk)
        require_ready(b)
        uid = b.video.user_id
        with ctx.locks(("user", uid)):
            try:
                out = _handle(ctx, session, b, body)
                session.commit()
                return out
            except ApiError:
                session.rollback()  # rejected before anything was applied
                raise
            except BaseException:
                session.rollback()
                ctx.policies.invalidate(uid)  # the learner may have been updated: rebuild it from the log next time
                raise


def _handle(ctx: AppContext, session: Session, b: Bundle, body: EventIn) -> StateOut:
    uid, pk = b.video.user_id, b.video.id
    events = _music(ctx, b, body) if body.kind == "music" else BUILDERS[body.kind](b, body)
    if not events:
        learner = ctx.policies.get(session, uid)
        return state_out(b, learner, message="No change: the edit already looks like that.")
    group = uuid.uuid4().hex[:10] if len(events) > 1 else None
    if group:
        for e in events:
            e.payload_json = {**e.payload_json, "group": group}
    # an explicit decision replaces the 'untouched' evidence recorded at finalize
    touched = {e.suggestion_id for e in events if e.suggestion_id is not None and e.kind in EXPLICIT}
    voided = [e for e in b.live if e.kind == "implicit" and e.suggestion_id in touched]
    if voided:
        for e in voided:
            e.undone = True
            session.add(e)
        session.flush()
        ctx.policies.invalidate(uid)
    learner = ctx.policies.get(session, uid)
    for e in events:
        session.add(e)
    _remember_defaults(session, b, events)
    session.flush()
    for e in events:
        if e.kind in LEARNED_KINDS:
            learner.apply_event(suggestion_from_row(b.sug_by_id[e.suggestion_id]), event_from_row(e))
    if b.video.status in ("ready", "exported") and body.kind != "rating":  # a verdict does not change the edit
        b.video.status = "edited"
    b.video.updated_at = time.time()
    session.add(b.video)
    session.flush()
    # re-score on what the new events changed
    changes = []
    b2 = load_bundle(ctx, session, pk)
    last = events[-1].id
    if any(e.kind in LEARNED_KINDS for e in events):
        changes = rescore(session, b2, learner, last, nudge=False)
    elif body.kind == "nudge":
        changes = rescore(session, b2, learner, last, nudge=True)
    session.flush()
    ctx.policies.commit(session, uid, learner)
    b3 = load_bundle(ctx, session, pk) if changes else b2
    msg = f"Adjusted {len(changes)} suggestion{'s' if len(changes) != 1 else ''} based on your last change" \
        if changes else None
    if body.kind == "nudge" and not changes:
        msg = "That nudge changed nothing: the untouched suggestions already lean that way."
    return state_out(b3, learner, changes=changes, message=msg, events=[event_dict(e) for e in events])


def _group_of(b: Bundle, e: EventRow, undone: bool) -> list[EventRow]:
    g = e.payload_json.get("group")
    if not g:
        return [e]
    return [x for x in b.ev_rows if x.payload_json.get("group") == g and x.undone == undone]


def _flip(b: Bundle, group: list[EventRow], undoing: bool) -> None:
    """Flip `undone` on the group and move the re-scoring side effects of those events with it."""
    ids = {e.id for e in group}
    for e in group:
        e.undone = undoing
    for s in b.sug_rows:
        if s.caused_by_event not in ids:
            continue
        old = b.sug_by_id.get(s.rescored_from) if s.rescored_from is not None else None
        if undoing:  # withdraw what the event created, bring back what it replaced
            s.superseded, s.superseded_by_event = True, s.caused_by_event
            if old is not None and old.superseded_by_event in ids:
                old.superseded, old.superseded_by_event = False, None
        else:
            s.superseded, s.superseded_by_event = False, None
            if old is not None:
                old.superseded, old.superseded_by_event = True, s.caused_by_event


def undo(ctx: AppContext, session: Session, video_pk: int) -> StateOut:
    return _flip_request(ctx, session, video_pk, undoing=True)


def redo(ctx: AppContext, session: Session, video_pk: int) -> StateOut:
    return _flip_request(ctx, session, video_pk, undoing=False)


def _flip_request(ctx: AppContext, session: Session, video_pk: int, undoing: bool) -> StateOut:
    with ctx.locks(("video", video_pk)):
        b = load_bundle(ctx, session, video_pk)
        require_ready(b)
        uid = b.video.user_id
        with ctx.locks(("user", uid)):
            try:
                target = undo_target(b) if undoing else redo_target(b)
                if target is None:
                    raise ApiError(409, "nothing to undo" if undoing else "nothing to redo")
                group = _group_of(b, target, undone=not undoing)
                _flip(b, group, undoing)
                for e in group:
                    session.add(e)
                for s in b.sug_rows:
                    session.add(s)
                session.flush()
                learner = ctx.policies.get(session, uid)  # fingerprint changed => rebuilt from the log
                ctx.policies.commit(session, uid, learner)
                b2 = load_bundle(ctx, session, video_pk)
                out = state_out(b2, learner, events=[event_dict(e) for e in group])
                session.commit()
                return out
            except ApiError:
                session.rollback()
                raise
            except BaseException:
                session.rollback()
                ctx.policies.invalidate(uid)
                raise
