"""Bring main.tex up to the final code. Text changes are applied line by line; results in @...@ come from paper/data.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\build_tex.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
BASE = P / "main_base.tex"
if not BASE.exists():
    BASE.write_text((P / "main.tex").read_text(encoding="utf-8"), encoding="utf-8")
lines = BASE.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
D = P / "data"
N = json.loads((D / "deep_numbers.json").read_text(encoding="utf-8"))
RC = json.loads((D / "real_clips.json").read_text(encoding="utf-8"))
NF = json.loads((D / "new_features.json").read_text(encoding="utf-8"))


def line(prefix: str, new: str) -> None:
    hits = [i for i, x in enumerate(lines) if x.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines[hits[0]] = new


def sub(old: str, new: str) -> None:
    hits = [i for i, x in enumerate(lines) if old in x]
    assert hits, old
    for i in hits:
        lines[i] = lines[i].replace(old, new)


def after(prefix: str, new: str) -> None:
    hits = [i for i, x in enumerate(lines) if x.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines.insert(hits[0] + 1, new)


def drop(prefix: str) -> None:
    hits = [i for i, x in enumerate(lines) if x.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    del lines[hits[0]]


# ---------------------------------------------------------------- title and abstract
TITLE = "Learning a Creator's Editing Style Online: Posterior Sampling With Style Priors, Uncertainty-Ordered Review, and Length Planning for Talking-Head Video"
line(r"\title{", r"\title{" + TITLE + "}")
line(r"\begin{abstract}" if False else "Short talking-head videos are cut by hand",
     "Short talking-head videos are cut by hand, and two creators rarely want the same cut from the same footage. Fixed rules cannot capture that difference, and supervised models need a labeled history that a new creator lacks. We treat each edit as a decision in online reinforcement learning, with the creator's accepts, rejections, and ordinary transcript edits as reward. A linear Thompson sampling agent handles the per-phrase decisions. Around it we add a tempered posterior over editing styles for the cold start, a review queue ordered by posterior uncertainty, an exact posterior-sampling planner for target lengths, and a randomized value-iteration agent for visual pacing that uses a delayed whole-video rating. Across six simulated creators and ten seeds, the method reached 91.0 percent acceptance by the tenth video, against 58.6 percent for a rule baseline, with 7 percent less regret than the same bandit without the style prior and 19 percent less than a neural agent. The planner cut held-out length error from 40.0 to 4.6 percent for 0.4 points of acceptance. Value iteration did not beat a bandit on pacing, and exploration raised no held-out score. Three recorded clips exercised the analysis pipeline, but no human study has been run.")

# ---------------------------------------------------------------- introduction
sub("We use the simplest formulation that captures this. Most editing decisions barely influence later contexts, so a contextual bandit \\cite{li2010,chu2011} is a defensible model, and it learns from a few hundred interactions where deep RL \\cite{mnih2015,schulman2017} would need many more. The sequential part of the problem, the length budget, gets a separate exact planner.",
    "We use the simplest formulation that captures each part. Most editing decisions barely influence later contexts, so a contextual bandit \\cite{li2010,chu2011} is a defensible model that learns from a few hundred interactions. The sequential parts, the length budget and the pacing of visual effects, get an exact planner and a randomized value-iteration agent. We tested the claim that deep RL \\cite{mnih2015,schulman2017} needs more data than this setting offers instead of assuming it.")
line(r"\item \textbf{A reward model in which", r"\item \textbf{A reward model in which ordinary editing is the training signal.} Word deletions, pause drags, and caption clicks are classified into overrides of learnable decisions, so a creator never performs a labeling step. In the ablation study, removing this mapping lowers early acceptance by 2.5 points.")
line(r"\item \textbf{A style posterior as", r"\item \textbf{A style posterior as a tempered bandit prior.} A Bayesian mixture over style prototypes supplies the prior mean of the bandit. Tempering lets the displayed style sharpen within about four interactions, whereas the prior moves slowly enough that one click cannot rewrite the policy. With leave-one-style-out priors, regret falls by 7 percent and the median number of decisions needed to reach 85 percent acceptance falls from 430 to 344.")
line(r"\item \textbf{Uncertainty-ordered review", r"\item \textbf{Uncertainty-ordered review under a budget.} Thompson draws give each suggestion a confidence (expected calibration error 0.023), and that value sorts the review queue. At a 30 percent review budget, 86.9 percent of exported decisions are correct, against 78.4 percent for random order and 63.3 percent for timeline order.")
line(r"\item \textbf{Posterior sampling for a target-length", r"\item \textbf{Posterior sampling for a target-length edit with guarded reward.} We pose length fitting as a finite-horizon Markov decision process (MDP) whose reward is the learned bandit model, solve it exactly, and show that the history features must be frozen to stop the solver from exploiting model extrapolation. Length error on unseen videos falls from 40.0 to 4.6 percent.")
line(r"\item \textbf{An explicit test of whether", r"\item \textbf{A measured account of what exploration and sequential models add here.} Policies trained with and without exploration differ by about one point on unseen videos, while exploration costs up to 5.7 points during training. On visual pacing with a delayed rating, randomized value iteration ties a bandit on the same features, and both match the creator better than tuned DQN and PPO agents at this data budget.")
line("Fig.~\\ref{fig:loop} shows the interaction", "Fig.~\\ref{fig:loop} shows the interaction at the conceptual level. The creator is part of the environment, the state is the feature vector of the current decision, and the policy update happens after each piece of feedback rather than in a separate training phase.")

# ---------------------------------------------------------------- related work
sub("Proximal policy optimization \\cite{schulman2017} was our first plan, but with known deterministic transitions an exact solution is both cheaper and optimal for the given reward.",
    "Proximal policy optimization \\cite{schulman2017} was our first plan, but with known deterministic transitions an exact solution is both cheaper and optimal for the given reward. Where the reward does depend on the state, as with pacing, randomized least-squares value iteration \\cite{osband2016,osband2019} explores coherently across an episode without a separate uncertainty model, and it is the closest sequential relative of the Thompson sampling we already use.")
sub("Riquelme \\emph{et al.} \\cite{riquelme2018} found that simple linear posteriors on good features remain hard to beat.",
    "Neural contextual bandits \\cite{zhou2020neural} replace the linear model with a network, although Riquelme \\emph{et al.} \\cite{riquelme2018} found that simple linear posteriors on good features remain hard to beat. Our results agree with that finding at this data scale.")
sub("We found no system that learns one creator's preferences online for this class of edit while also planning to a length target and logging what is needed for counterfactual evaluation.",
    "We found no system that learns one creator's preferences online for this class of edit while also planning to a length target and logging what is needed for counterfactual evaluation, although our search was not exhaustive.")

# ---------------------------------------------------------------- method
line("Our system, RoughCut, turns raw speech", "Our system, RoughCut, turns raw speech footage into a set of editing decisions, proposes an action for each with per-creator agents, and applies all proposals straight away so that the creator starts from a finished first cut. Corrections then flow back through a reward mapping. What sets it apart is what surrounds the bandit: the style posterior used as its prior, the review queue built from its uncertainty, an exact posterior-sampling planner for target lengths, a randomized value-iteration agent for pacing, and propensity logging for counterfactual evaluation. Fig.~\\ref{fig:pipeline} shows the full pipeline.")
sub("Audio is transcribed with faster-whisper running the small English Whisper model \\cite{radford2023}, without conditioning on previous text because conditioning caused looping output over silence.",
    "Audio is transcribed with faster-whisper running the small English Whisper model \\cite{radford2023} at beam size 10, without conditioning on previous text because conditioning caused looping output over silence. The decoder runs in its own process: inside a process that had already prepared the footage, identical audio produced 305 words instead of 324 on one clip and lost a whole sentence. We did not find the internal cause, but three repeated runs in fresh processes were identical.")
after("Two steps go beyond that standard recipe.", "Spoken lines can also be aimed at the wrong audience. A creator may say ``this one is for YouTube Shorts'' to a colleague, and that line is clean, complete English, so filler and retake detection pass it. Each sentence therefore gets a crew score. Words about production (``voiceover'', ``one for Instagram'') add 0.4 and softer signals add 0.2. Only when such words are present do quiet delivery, isolation between long gaps, and shortness add to the score. Viewer-directed words (``comment'', ``subscribe'') subtract from it, and a score of at least 0.5 cuts the line in the first edit. The creator can restore it like any other cut. These weights were set by hand from very few clips, so we treat the rule as a useful default and not as a validated detector.")
line("Each decision has a context vector", "Each decision has a context vector $x_t \\in \\mathbb{R}^{d}$, normalized and partly binned, with $d$ between 16 and 39 depending on its type (Table~\\ref{tab:decisions}). Contexts also carry a one-hot video type and a bias term. Decisions are processed left to right, so the context can include history features such as the ratio of output time to source time so far and the time since the last visual change. In the bandit layer these features are the only way an earlier action affects a later state.")
line("Emphasis & NONE, or one of", "Emphasis & NONE, or one of up to 3 words & 16 & per word: number, proper noun, TF-IDF, stress, negation \\\\\nPace & S100, S110, S125 & 18 & speech rate, duration, loudness, filler ratio \\\\\nMusic & FROM\\_START, AFTER\\_INTRO & 7 & video length, first-sentence length, type \\\\\nBeat & NONE, Z115, Z130, TEXT\\_POP, SFX & 29 & time since last beat, its kind, streak, centrality \\\\\nTitle & one of up to 3 lines & 8 & first line, question, number, ``you'', centrality \\\\")
line("Phrase, pause, zoom, and filler decisions use", "Phrase, pause, filler, pace, and music decisions use disjoint arms, with one weight vector per action. Caption emphasis uses a shared weight vector over per-word features, so what is learned about numbers on one line transfers to all lines. Pause actions leave 0.40 s, 0.15 s, or 0.03 s of silence, or keep the gap untouched. All decisions compile into a single edit decision list, the only representation that both the browser preview and the FFmpeg renderer read. Speech edges are padded by 40 ms, every cut has an 8 ms audio fade, and clip edges snap to the 30 frames per second grid so that the rendered length equals the planned length. The transition dynamics are deterministic and known: applying an action changes only the running output time and history counters. The simulator experiments use a zoom decision with the first three beat actions; the deployed editor replaces it with the beat decision of Section~\\ref{sec:rhythm}.")
sub("Undo or redo & flip event flag, replay log \\\\", "Rating at export, $\\rho\\in\\{-1,0,1\\}$ & rhythm agent only (Section~\\ref{sec:rhythm}) \\\\\nUndo or redo & flip event flag, replay log \\\\")
sub("product, since a uniform draw there can delete a sentence", "product, since a uniform draw there can delete a sentence")
sub("It finishes in about one second on a laptop CPU", "It finishes in about one second on a laptop CPU") if False else None
sub("and it finishes in about one second on a laptop CPU (Table~\\ref{tab:runtime})", "and it finishes in about 0.2 s on a laptop CPU (Table~\\ref{tab:runtime})")
line("\\subsubsection{Counterfactual Evaluation}", r"""\subsubsection{Visual Rhythm by Randomized Value Iteration}
\label{sec:rhythm}
Whether a punch-in works depends on what came before it, so this decision is sequential by nature. At each kept phrase longer than 1 s, the agent chooses among no effect, a 1.15x or 1.3x zoom, a text pop-up, and a sound effect. The state $\phi(s)\in\mathbb{R}^{29}$ holds binned time since the last beat, the kind of that beat, the streak length, beats per minute so far, phrase features, and products of time bins with topic centrality, because the interesting rules are conjunctions. Unlike the planner's frozen features, these are true state: the chosen action changes the next $\phi$.

With a linear $Q(s,a)=\theta_a^{\top}\phi(s)$, RLSVI \cite{osband2016} perturbs each reward with Gaussian noise and each prior with a draw, and then runs fitted value iteration on the perturbed data,
\begin{equation}
y_i = r_i + \xi_i + \gamma \max_{a'} \theta_{a'}^{\top}\phi(s_i'), \qquad \xi_i \sim \mathcal{N}(0,\sigma^{2}/w_i), \label{eq:rlsvi}
\end{equation}
\begin{equation}
\theta_a \leftarrow \Big(\sum_{i:a_i=a} w_i\phi_i\phi_i^{\top} + \lambda I\Big)^{-1}\Big(\sum_{i:a_i=a} w_i\phi_i y_i + \lambda\tilde\theta_a^{0}\Big), \label{eq:rlsvi2}
\end{equation}
with $\gamma=0.9$, $\sigma=0.15$, $\lambda=1$, and 10 sweeps. The agent then acts greedily on that one draw for the whole video, which is Thompson sampling extended to an episode. The prior mean $\theta^{0}$ is fitted to a calm hand-written rule, one modest beat after several quiet seconds. The rating at export is delayed feedback: it adds $2\rho$ to the last step, for value iteration to carry backward, and $0.5\rho$ to every step. Those two weights were chosen on tuning seeds only. Algorithm~\ref{alg:rlsvi} gives the procedure, and setting $\gamma=0$ turns the same code into a bandit on the same features, which is the baseline we compare with.

\begin{algorithm}[t]
\caption{Visual Rhythm Agent (RLSVI)}
\label{alg:rlsvi}
\begin{algorithmic}[1]
\small
\REQUIRE Prior $\theta^{0}$, logged episodes $\mathcal{D}$
\FOR{each new video}
\STATE Perturb prior: $\tilde\theta^{0}\leftarrow\theta^{0}+\sigma\lambda^{-1/2}\varepsilon$
\STATE Perturb rewards in $\mathcal{D}$; add the rating to steps
\FOR{10 sweeps}
\STATE Compute targets by \eqref{eq:rlsvi}; update $\theta_a$ by \eqref{eq:rlsvi2}
\ENDFOR
\FOR{each kept phrase in time order}
\STATE Observe $\phi(s)$; take $a=\arg\max_a \theta_a^{\top}\phi(s)$
\STATE Observe the creator's reaction; advance the state
\ENDFOR
\STATE Observe the rating $\rho$; append the episode to $\mathcal{D}$
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsubsection{Opening Title, Pace, and Music Timing}
Three smaller decisions use the same machinery. A hook bandit picks the on-screen title from up to three candidate lines, using a shared Thompson model over line features, so a preference for questions carries across videos. It does not move footage to the front. Phrase speed and music timing are ordinary LinTS decisions with a fixed bonus on the plain option, so an untrained agent stays at normal speed. Cosmetic decisions are capped in the review queue, since they are numerous and cheap to get wrong. We do not evaluate these three decisions in the simulator, so they are product features here and not research results.

\subsubsection{Counterfactual Evaluation}""")
sub("Algorithm~\\ref{alg:main} gives the online loop and Algorithm~\\ref{alg:plan} the planner.", "Algorithm~\\ref{alg:main} gives the online loop, Algorithm~\\ref{alg:plan} the planner, and Algorithm~\\ref{alg:rlsvi} the rhythm agent.")
sub("More than 300 automated backend tests cover the reward mapping, replay, and timing rules, and a further 160 cover the interface.", "The 317 automated backend tests cover the reward mapping, replay, and timing rules, and a further 166 cover the interface.")

# ---------------------------------------------------------------- setup
sub("We also use two recorded clips for the analysis layer.", "We also use three recorded clips for the analysis layer.")
sub("Agents are our own implementation; Gymnasium and Stable Baselines3 are not used because the planner is solved exactly.", "Agents are our own implementation, and the deep baselines are written directly in PyTorch. Gymnasium and Stable Baselines3 are not used, because the environment runs the product's own decision code.")
line("Posterior scale $v$ & 0.25 & 0.25", "Posterior scale $v$ & 0.25 & 0.15 \\\\")
line("Exploration floor $\\epsilon$ & 0.02", "Exploration floor $\\epsilon$ & 0.02 & 0.005 (keep/cut 0) \\\\")
line("Episodes per run / seeds per persona", "Episodes per run / seeds per persona & \\multicolumn{2}{c|}{10 / 10} \\\\\nRLSVI $\\gamma$, $\\sigma$, $\\lambda$, sweeps & \\multicolumn{2}{c|}{0.9, 0.15, 1, 10} \\\\\nNeural bandit: hidden, batch, lr, $\\epsilon$ & \\multicolumn{2}{c|}{64--64, 32, $10^{-3}$, 0.05} \\\\")
sub("The terms learning rate and batch size from deep RL have no direct counterpart here.", "For the linear agents, the terms learning rate and batch size have no direct counterpart.")

# ---------------------------------------------------------------- performance
sub("whereas Rule makes 670.2 and Global 780.5.", "whereas Rule makes 668.9 and Global 780.1.")
sub("The proposed method ends at 90.7 percent acceptance with 184.5 wrong proposals in 10 videos", "The proposed method ends at 91.0 percent acceptance with 181.5 wrong proposals in 10 videos")
d = N["decisions"]
ne, np_ = d["Neural"], d["Neural, pretrained"]
ROWS = {
    "Rule (one-click tool)": r"Rule (one-click tool) & 0.594 & 0.596 & 0.586 $\pm$ 0.031 & 0.408 & 0.000 & never & 668.9 $\pm$ 45.0 & 19.91 \\",
    "Global prior, no learning": r"Global prior, no learning & 0.525 & 0.513 & 0.521 $\pm$ 0.035 & 0.306 & 0.000 & never & 780.1 $\pm$ 46.5 & 22.77 \\",
    "$\\epsilon$-greedy": r"$\epsilon$-greedy & 0.777 & 0.862 & 0.874 $\pm$ 0.011 & 0.769 & 0.700 & 519.0 & 232.8 $\pm$ 14.1 & 7.64 \\",
    "LinUCB &": r"LinUCB & 0.779 & 0.873 & 0.903 $\pm$ 0.011 & 0.809 & 0.823 & 517.5 & 208.1 $\pm$ 13.0 & 6.66 \\" + "\n" +
        f"Neural $\\epsilon$-greedy, blank & {ne['v1']:.3f} & {ne['v5']:.3f} & {ne['v10']:.3f} $\\pm$ {ne['v10_ci']:.3f} & -- & {ne['success']:.3f} & {ne['steps_to_85']:.1f} & {ne['regret']:.1f} $\\pm$ {ne['regret_ci']:.1f} & -- \\\\\n" +
        f"Neural $\\epsilon$-greedy, pretrained & {np_['v1']:.3f} & {np_['v5']:.3f} & {np_['v10']:.3f} $\\pm$ {np_['v10_ci']:.3f} & -- & {np_['success']:.3f} & {np_['steps_to_85']:.1f} & {np_['regret']:.1f} $\\pm$ {np_['regret_ci']:.1f} & -- \\\\",
    "LinTS, generic prior &": r"LinTS, generic prior & 0.790 & 0.891 & 0.904 $\pm$ 0.011 & 0.814 & 0.853 & 430.0 & 194.7 $\pm$ 14.7 & 6.27 \\",
    "\\textbf{LinTS, style prior (proposed)}": r"\textbf{LinTS, style prior (proposed)} & \textbf{0.806} & 0.890 & \textbf{0.910 $\pm$ 0.012} & 0.807 & \textbf{0.903} & \textbf{344.0} & \textbf{181.5 $\pm$ 11.0} & 6.19 \\",
    "Proposed, product settings &": r"Proposed, product settings & 0.825 & 0.900 & 0.912 $\pm$ 0.012 & 0.821 & 0.907 & 343.5 & 168.4 $\pm$ 11.7 & 5.82 \\",
}
for k, v in ROWS.items():
    hits = [i for i, x in enumerate(lines) if x.startswith(k) and " & " in x and x.rstrip().endswith("\\\\") and x.count("&") == 8]
    assert len(hits) == 1, k
    lines[hits[0]] = v
sub("Pauses are where rules do worst, at 0.42 on video 10, because the personas disagree most on them, and they are where learning gains most (0.96). Keep/cut starts high for every method because most phrases should be kept, and here the Rule baseline's 0.87 is close to the learners. Emphasis is the hardest learned decision at 0.86",
    "Pauses are where rules do worst, at 0.42 on video 10, because the personas disagree most on them, and they are where learning gains most (0.96). Keep/cut starts high for every method because most phrases should be kept, and here the Rule baseline's 0.87 is close to the learners. Emphasis is the hardest learned decision at 0.88")
line("Fig.~\\ref{fig:persona} and Table~\\ref{tab:persona} repeat", "Fig.~\\ref{fig:persona} and Table~\\ref{tab:persona} repeat the comparison per creator. The proposed method is ahead of Rule for all six, by 19 to 50 points. The gap is smallest for Blend and Snappy, whose pause rule resembles the one in the Rule baseline, and largest for Relaxed and Minimalist, whom a fixed cut-everything-over-half-a-second policy serves poorly. Against plain LinTS the style prior helps most on Snappy and Blend. On Minimalist and Focused it is slightly behind (0.857 against 0.870, and 0.915 against 0.929), so the mean improvement is not a uniform one.")
for k, v in {"Snappy & 0.703": r"Snappy & 0.701 & 0.644 & 0.881 & 0.868 & \textbf{0.893} \\", "Relaxed & 0.472": r"Relaxed & 0.472 & 0.348 & 0.956 & 0.961 & \textbf{0.974} \\",
             "Storyteller & 0.673": r"Storyteller & 0.671 & 0.600 & \textbf{0.912} & 0.906 & 0.910 \\", "Minimalist & 0.419": r"Minimalist & 0.422 & 0.346 & 0.860 & \textbf{0.870} & 0.857 \\",
             "Focused & 0.529": r"Focused & 0.531 & 0.535 & \textbf{0.933} & 0.929 & 0.915 \\", "Blend & 0.725": r"Blend & 0.721 & 0.652 & 0.875 & 0.887 & \textbf{0.910} \\"}.items():
    line(k, v)
line("Table~\\ref{tab:type} gives the same breakdown", "Table~\\ref{tab:type} gives the same breakdown by decision type. The product settings, which use a stronger keep/cut prior, trade about three points of keep/cut acceptance (0.876 against 0.905) for protection against sudden large changes, and they recover the difference on pauses and zooms. This is a deliberate choice. A wrong CUT removes what someone said, whereas a wrong pause length is hardly noticed, so we accept lower simulated keep/cut acceptance in exchange for slower movement on that one decision type.")
for k, v in {"Rule & 0.869": r"Rule & 0.869 & 0.416 & 0.739 & 0.745 & 0.433 \\", "Global prior & 0.843": r"Global prior & 0.847 & 0.337 & 0.633 & 0.593 & 0.403 \\", "LinUCB & 0.900": r"LinUCB & 0.907 & 0.941 & 0.967 & 0.945 & 0.871 \\",
             "LinTS, generic prior & 0.894": r"LinTS, generic prior & 0.904 & 0.953 & 0.931 & 0.963 & 0.868 \\", "Proposed & 0.917": r"Proposed & 0.905 & 0.961 & 0.931 & 0.961 & 0.883 \\", "Product settings & 0.885": r"Product settings & 0.876 & 0.983 & 0.943 & 0.973 & 0.889 \\"}.items():
    line(k, v)
line("Fig.~\\ref{fig:seeds} shows the spread", "Fig.~\\ref{fig:seeds} shows the spread for individual creators. The standard deviation of per-seed mean acceptance at video 10 is 0.013 for the proposed method and 0.018 for the plain bandit, both small compared with the 0.32 gap from Rule. Most of the spread across all runs (0.045) comes from differences among personas and not from randomness, since Relaxed reaches 0.974 and Minimalist only 0.857. Minimalist's rule depends on a per-video ranking of centrality that a linear model cannot fully represent.")
line("Table~\\ref{tab:runtime} shows the learner's cost", "Table~\\ref{tab:runtime} shows the learner's cost, measured on one thread. A suggestion takes well under 1 ms on average, so it is the media analysis and not the learning that determines waiting time. The bandit has 305 parameters per creator and its data statistics occupy 67 KiB. The rhythm agent adds 145 parameters and takes @RLSVI_MS@ ms to prepare a video after ten videos of history. A neural bandit run took @NEURAL_S@ s against 2.4 s for LinTS. Analysis took 98 s for the 54.5 s clip, 119 s for the 44.2 s clip, and 270 s for the 238.5 s clip, measured while the test suite was running on the same machine, so these are upper figures.")
line("Thompson choice with 200 draws (ms)", "Thompson choice with 200 draws (ms) & 0.07 & 0.10 \\\\")
line("Update of one arm (ms)", "Update of one arm (ms) & 0.02 & 0.03 \\\\")
line("Plan one video, 65 stages (ms)", "Plan one video, 65 stages (ms) & 186 & 197 \\\\")
line("Simulated run, 10 videos, 1600 decisions (s)", "Simulated run, 10 videos, 1600 decisions (s) & \\multicolumn{2}{c|}{2.4} \\\\")

# ---------------------------------------------------------------- discussion
line("On videos 1 to 3, acceptance is 0.823", "On videos 1 to 3, acceptance is 0.824 with no prior, 0.831 with a generic prior, and 0.845 with the style mixture. The improvement is real but modest, and it fades as expected: by video 6 the three curves lie within one point of each other (0.889, 0.889, 0.899). Personal data eventually dominates any prior. What the prior buys is time. The median number of decisions to reach an 85 percent video drops from 430 to 344, about half a video sooner. The style itself is identified (weight above 0.8) after a median of 3.5 events, although Snappy takes 7.5 and at most 21, as it overlaps with Blend. These numbers come from leave-one-style-out priors, so they probably describe a new creator somewhat similar to the known styles, and a person who resembles none of them should expect less.")
line("Fig.~\\ref{fig:cold} shows how the gap closes", "Fig.~\\ref{fig:cold} shows how the gap closes. The lead of the style mixture is largest on the first video (0.806 against 0.767 with no prior) and is nearly gone by the fourth, where all three are near 0.885. A paired test on the first three videos gives a gain of 1.5 points over the generic prior ($p=0.004$). In practical terms, this is about 13 fewer wrong proposals across ten videos against the generic prior, and it costs almost nothing to compute: one mixture update per event.")
sub("The diagonal averages 0.940 and the off-diagonal 0.466", "The diagonal averages 0.936 and the off-diagonal 0.468")
line("Fig.~\\ref{fig:drift} shows a creator who switches", "Fig.~\\ref{fig:drift} shows a creator who switches from Relaxed to Snappy at video 6. Acceptance collapses to 0.54 with no discounting, 0.61 with $\\gamma=0.995$, and 0.68 with $\\gamma=0.98$, and all three recover above 0.8 within three to four videos. Stronger discounting softens the immediate drop but pays for it, with lower acceptance before the switch (0.935 against 0.950) and at the end of the run (0.834 against 0.871). Discounting costs accuracy in stationary conditions, as the ablation also shows, so the choice of $\\gamma=0.995$ is a compromise rather than an optimum.")
line("Fig.~\\ref{fig:target} compares the bandit", "Fig.~\\ref{fig:target} compares the bandit and the bandit with planner, both trained online and then frozen on four unseen videos, for targets of 20, 30, and 45 s. The bandit's length error remains 40.0 percent after six videos, because nothing in a per-decision policy knows the target. With the planner, the error ends at 4.6 percent, below the 5 percent tolerance, and it is small from the first video onward because the target is enforced by planning rather than learned. It costs 0.4 points of acceptance (0.879 against 0.883), which lies within the intervals. Keep/cut acceptance stays near 0.68 for both, so neither reproduces Minimalist's choice of which phrases to keep. Planning fixes the length without fixing the selection, and this is the method's clearest unsolved problem.")
sub("with 59.8 percent error at 20 s", "with 54.7 percent error at 20 s")
line("Bandit only & 0.598", "Bandit only & 0.547 & 0.340 & 0.314 \\\\")
line("Bandit with planner & 0.042", "Bandit with planner & 0.054 & 0.052 & 0.033 \\\\")
sub("whereas the planner stays under 6 percent at all three", "whereas the planner stays under 6 percent at all three")
line("Fig.~\\ref{fig:review} varies the share", "Fig.~\\ref{fig:review} varies the share of decisions a creator is willing to review. At 30 percent, an uncertainty-ordered queue gives 0.869 correct decisions in the export, against 0.784 for random order and 0.633 for timeline order. At full review, all three are the same as they must be, at about 0.90. The gain comes from catching errors: confidence predicts mistakes well enough that the same effort fixes more of them. Policies trained at a 30 percent budget also score higher when frozen after uncertainty-ordered review (0.841 against 0.800), which suggests that the queue selects useful training samples too, though the difference is near the width of the intervals.")
line("The queue only works if confidence", "The queue only works if confidence tracks mistakes, so we measured that directly (Fig.~\\ref{fig:calib}, Table~\\ref{tab:calib}). At $v=0.25$ the expected calibration error over about 58\\,000 suggestions is 0.023. Suggestions below the 0.7 threshold make up 10.1 percent of the total and are right only 37.1 percent of the time, whereas those above it are right 94.4 percent of the time. The area under the curve for telling right from wrong is 0.89. So a creator who looks only at the flagged tenth is looking where most errors are.")
line("This is what a smaller $v$ loses.", "This is what a smaller $v$ loses. At $v=0.05$, acceptance is higher (0.916 against 0.902 late in training), but the area under the curve falls to 0.72 and only 3.2 percent of suggestions fall below the threshold, so many mistakes go unflagged. At $v=0.5$ the ranking is sharper (0.94), but 18.6 percent are flagged and acceptance falls to 0.876. The experiments use $v=0.25$ and the product $v=0.15$, chosen between those extremes: on 120 fresh runs the product's values reached 0.903 against 0.892 after six videos (paired difference 0.011 $\\pm$ 0.005). This result also gives a more accurate account of what Thompson sampling contributes in this system. It does not seem to help learning under our feedback model, but a posterior of the right width produces an uncertainty signal that a creator can use.")
for k, v in {"Expected calibration error &": r"Expected calibration error & 0.065 & 0.023 & 0.023 \\", "Area under the curve (right": r"Area under the curve (right vs. wrong) & 0.715 & 0.893 & 0.940 \\", "Share flagged (confidence": r"Share flagged (confidence $<0.7$) & 0.032 & 0.101 & 0.186 \\",
             "Correct among flagged &": r"Correct among flagged & 0.296 & 0.371 & 0.372 \\", "Correct among not flagged &": r"Correct among not flagged & 0.919 & 0.944 & 0.963 \\", "Late acceptance (videos 8--10)": r"Late acceptance (videos 8--10) & 0.916 & 0.902 & 0.876 \\"}.items():
    line(k, v)
line("Fig.~\\ref{fig:ope} and Table~\\ref{tab:ope} evaluate", "Fig.~\\ref{fig:ope} and Table~\\ref{tab:ope} evaluate policies from LinTS logs, with simulator truth as reference. On average the doubly robust estimate ranks the learned policy first by a wide margin and comes within 1.3 points of it (true 0.932, DR 0.945), and it recovers the order of Rule (0.749, DR 0.739) and the generic prior (0.703, DR 0.694). DM has the lowest error (RMSE 0.065) but its bias makes its intervals almost useless, covering the truth only 48 percent of the time. DR has much smaller bias (0.002) and the best coverage, 91 percent. The small-sample $t$ correction raised coverage from the 71 to 78 percent we saw with percentile intervals, but it still falls short of the nominal 95 percent. Part of the shortfall is likely adaptive logging \\cite{hadad2021}. The effective sample size is just 15 percent of the 950 logged decisions per creator, because LinTS quickly becomes confident and stops supporting other actions. Offline evaluation in this setting therefore works for ranking clearly different policies, but not for detecting small differences.")
for k, v in {"DM & 0.040": r"DM & 0.040 & 0.065 & 0.48 & 1.00 \\", "IPS & 0.009": r"IPS & $-$0.007 & 0.153 & 0.72 & 0.72 \\", "SNIPS & 0.020": r"SNIPS & 0.013 & 0.124 & 0.81 & 0.97 \\", "DR & 0.007": r"DR & 0.002 & 0.126 & 0.91 & 0.94 \\", "SNDR & 0.009": r"SNDR & 0.003 & 0.121 & 0.86 & 0.94 \\"}.items():
    line(k, v)

r, a, s = RC["real"], RC["aroll"], RC["scripted"]
flag = lambda c: c["low_confidence_decisions"]  # noqa: E731
line("Simulation says nothing about whether", "Simulation says nothing about whether the speech analysis works, so we ran three clips through the actual pipeline and compared the first cut a brand-new creator would receive with a reference edit (Table~\\ref{tab:real}). The first is a 44.2 s recording of a person with a real chain of restarts: four abandoned attempts at one sentence followed by the clean take, and unrelated background speech in a 7 s gap. The first cut drops every abandoned attempt. The second is a 238.5 s recording with long gaps, six retaken lines, and five phrases spoken to someone off camera. All five were cut and the call to action at the end was kept. The third is a 54.5 s scripted stress clip with a synthetic voice, containing fillers, a stutter, hedges, a three-attempt restart, and a sentence where the first delivery is the good one. The agent resolved all of them and retained one filler (``basically''), which has no audio dip around it.")
sub("\\caption{First Cut Against a Hand-Made Reference on Recorded Clips}", "\\caption{First Cut Against a Reference Edit on Recorded Clips}")
sub("\\begin{tabular}{|p{37mm}|c|c|}", "\\begin{tabular}{|p{33mm}|c|c|c|}")
line("Measure & Real speaker &", "Measure & Clip A & Clip B & Stress clip \\\\")
line("Source length (s) & 44.2", f"Source length (s) & {r['source_s']:.1f} & {a['source_s']:.1f} & {s['source_s']:.1f} \\\\")
line("First cut length (s) & 22.1", f"First cut length (s) & {r['first_cut_s']:.1f} & {a['first_cut_s']:.1f} & {s['first_cut_s']:.1f} \\\\")
line("Clips in the edit & 5", f"Clips in the edit & {r['clips']} & {a['clips']} & {s['clips']} \\\\")
line("Suggestions / flagged for review", f"Suggestions / confidence under 0.7 & {r['n_suggestions']} / {flag(r)} & {a['n_suggestions']} / {flag(a)} & {s['n_suggestions']} / {flag(s)} \\\\\nRetake phrases cut & {r['retake_phrases_cut']} & {a['retake_phrases_cut']} & {s['retake_phrases_cut']} \\\\\nCrew phrases cut & {len(r['crew_lines_cut'])} & {len(a['crew_lines_cut'])} & {len(s['crew_lines_cut'])} \\\\")
line("Reference words & 94", f"Reference words & {r['ref_words']} & {a['ref_words']} & {s['ref_words']} \\\\")
line("Word differences (word error rate)", f"Word differences from reference & {r['word_edits']} & {a['word_edits']} & {s['word_edits']} \\\\")
sub("\\multicolumn{3}{p{80mm}}{Differences are measured on caption words. The reference keeps Whisper's spelling of the speaker's name, since a spelling error is not an editing error.}",
    "\\multicolumn{4}{p{80mm}}{Differences are measured on caption words. References keep the transcript's wording, so this checks which lines survive, not transcription accuracy.}")
line("These are encouraging results but very thin evidence.", "These are encouraging results but very thin evidence. The sample is three clips and one of them has a synthetic voice. The references were prepared with AI assistance from the transcript and from decoding doubtful stretches on their own, not by an independent editor listening, and they use the transcript's wording, so zero differences does not mean perfect transcription: the system writes ``Klein'' for ``client'', and no reference corrects it. The crew rules and the process isolation were developed on these same clips, so they are not held out. Through this comparison we found and fixed a stuttered scrap of a restart and the lost sentence described earlier. There are certainly more such problems in footage we have not tried. Nobody has yet reviewed the rendered audio by ear.")
line("Acceptance alone does not show whether", "Acceptance alone does not show whether the learned weights make sense, so we queried the posterior means directly. We trained with no prior, for ten videos and six seeds, so that everything comes from feedback, and then scored hand-built contexts (Table~\\ref{tab:learned}). The sign of each entry says which action the model favors. The results read like the personas' hidden rules. All three creators remove ``um'', but only Snappy removes ``like'', and the others keep it. A 0.6 s pause is removed by Snappy and kept firmly by Relaxed, and Relaxed nearly flips between 0.6 s and 1.5 s, which is the threshold in its rule. An earlier take is cut and an ordinary phrase is kept by all, the behavior shared by every style. This is the basis of the profile page, and it also shows a benefit of keeping the model linear: such a check takes a few lines of code.")
for k, v in {"``um'': REMOVE": r"``um'': REMOVE $-$ KEEP & +0.94 & +0.95 & +0.49 \\", "``like'': REMOVE": r"``like'': REMOVE $-$ KEEP & +0.91 & $-$0.49 & $-$0.89 \\", "Pause 0.6 s: REMOVE": r"Pause 0.6 s: REMOVE $-$ KEEP & +0.88 & $-$0.78 & +0.38 \\",
             "Pause 1.5 s: REMOVE": r"Pause 1.5 s: REMOVE $-$ KEEP & +1.09 & +0.07 & +0.48 \\", "Earlier take: CUT": r"Earlier take: CUT $-$ KEEP & +0.63 & +0.39 & +0.60 \\", "Ordinary phrase: CUT": r"Ordinary phrase: CUT $-$ KEEP & $-$0.69 & $-$1.01 & $-$0.73 \\"}.items():
    line(k, v)

# ---------------------------------------------------------------- comparison
rh = N["rhythm"]
vp = ne["vs_proposed"]
vq = np_["vs_proposed"]
RH = "\n".join(f"{n} & {rh[n]['v3']:.3f} & {rh[n]['final']:.3f} $\\pm$ {rh[n]['final_ci']:.3f} & {rh[n]['rating']:+.2f} $\\pm$ {rh[n]['rating_ci']:.2f} \\\\".replace("RLSVI &", "\\textbf{RLSVI} &")
               for n in ("Fixed rule", "Linear bandit", "RLSVI", "DQN, blank", "DQN, warm start", "PPO, blank", "PPO, warm start"))
line("Table~\\ref{tab:compare} is the main benchmark.", r"""Table~\ref{tab:compare} is the main benchmark. Compared with the Rule baseline, modeled on what one-click tools do, the proposed method cuts regret by 73 percent and corrections per minute of footage by 69 percent, from 19.9 to 6.2. Against the unadapted Global prior the gap is wider still, which shows that a pooled model is worse than a simple rule when styles disagree. Among learners, the differences are much smaller. Video 10 acceptance for LinUCB (0.903) and proposed (0.910) overlaps within intervals. A paired comparison on matched persona and seed puts the regret advantage of the proposed method at 26.6 $\pm$ 9.1 decisions over LinUCB and 13.3 $\pm$ 9.3 over plain LinTS, both with $p<0.01$. On video 10 alone it is not distinguishable from either ($p=0.15$ and $p=0.22$). No correction for multiple comparisons was applied.

\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)[width=\textwidth]{figures/deep.pdf}
{\parbox[t]{0.91\textwidth}{\raggedright Neural agents against the linear methods on the same creators and seeds. The neural agents start lower and catch most of the gap by the tenth video, so their cost is the regret paid on the way there.}\label{fig:deep}}

We also ran a deep baseline, since the case for a linear model should rest on a measurement. The neural agent has one 64-by-64 network per decision type, trained online from a replay buffer with $\epsilon$-greedy actions, which is the DQN recipe \cite{mnih2015} with $\gamma=0$ because these decisions have no transition. Learning rate and $\epsilon$ were chosen on separate seeds. Fig.~\ref{fig:deep} shows the result, and it is closer than we expected. By video 10 the blank network reaches @N_V10@, only @N_DV10@ points behind the proposed method. The gap is in the early videos: @N_V1@ on the first video against 0.806, and @N_DR@ more wrong proposals over ten videos. Pretraining on the other creators' labels, the same information the generic prior has, helps less than the prior does, with @NP_V1@ on the first video and @NP_DR@ more wrong proposals. A linear model is therefore justified here by early performance, cost, exact replay, and readable explanations. It is not justified by a higher final score, and a longer horizon might favor the network.

\subsubsection{Visual Rhythm: Is a Sequential Agent Needed?}
Table~\ref{tab:rhythm} and Fig.~\ref{fig:rhythm} test the sequential agent on four simulated creators whose preferences depend on history: one modest punch-in after a quiet stretch, a strong beat only on important phrases after eight quiet seconds, a beat every three seconds without repeating the kind, and almost nothing. Each edits eight videos with feedback on every beat and rates each whole video by how close its beats per minute are to what the creator likes. Scores are on four unseen videos, over 20 seeds not used for tuning. DQN and PPO were tuned over learning rate and $\epsilon$ or update epochs on separate seeds, from a blank start and from a start fitted to the same rule that gives RLSVI its prior.

\begin{table}[t]
\caption{Visual Rhythm on Unseen Videos (Four Creators, 20 Seeds)}
\label{tab:rhythm}
\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{|l|c|c|c|}
\hline
Agent & Match, video 3 & Match, video 8 & Rating, video 8 \\
\hline
""" + RH + r"""
\hline
\end{tabular}
\end{table}

\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)[width=\textwidth]{figures/rhythm.pdf}
{\parbox[t]{0.91\textwidth}{\raggedright Visual rhythm on unseen videos. (a) Share of beats that match the creator's ideal. (b) Whole-video rating, in $[-1,1]$. Bands show 95 percent intervals over 80 runs per agent.}\label{fig:rhythm}}

RLSVI did not beat the bandit. Its final match is 0.901 against 0.919 (paired difference $-$0.018 $\pm$ 0.019, $p=0.055$), and their ratings are level at +0.04. We expected the sequential agent to win, and the likely reason it did not lies in our design. Every state these creators react to is also visible to the bandit through the history features, so choosing the best action for the current state is already close to optimal. The delayed rating adds little that per-beat feedback has not already said. The largest difference is on the minimal creator (0.816 against 0.894), where the agent chases rare positive beats. Both agents still fail on rating for the two creators who want few beats, because a few wrong beats are enough to miss a narrow target rate. @DEEP_TEXT@ A fair test of RLSVI needs creators who give mostly delayed feedback, and we have not run one.""")

# ---------------------------------------------------------------- ablation
line("Table~\\ref{tab:ablation} and Fig.~\\ref{fig:ablation} remove", "Table~\\ref{tab:ablation} and Fig.~\\ref{fig:ablation} remove one component at a time, with paired seeds. Three components make clear positive contributions. Manual-edit mapping adds 2.5 points on early videos and 1.1 late, as half of the corrections would otherwise be lost. Retake features add 2.1 points early and 1.6 late, the largest lasting effect, as expected since every persona cuts earlier takes. The style prior adds 1.5 points early and 0.8 late. Removing the prior altogether gives the largest early drop among prior variants (2.1) with no significant late penalty. Semantic and history features have almost no effect in the simulator. For semantic features that may be partly an artifact, because synthetic embeddings are only crude stand-ins for language.")
line("Three removals do not hurt", "Three removals do not hurt, and two of them improve the score, which runs against the usual narrative. Removing implicit feedback changes little (+0.2 late, within noise). Removing discounting adds 1.8 points late, which is expected since these personas do not drift, and Fig.~\\ref{fig:drift} shows the opposing case. The more important result is exploration: a policy with $v=0$ and $\\epsilon=0$ scores 2.7 points higher early and 2.6 later, with regret of 140.8 against 181.5.")
for k, v in {"Nothing (full method)": r"Nothing (full method) & 0.845 & 0.903 & 181.5 \\", "Manual-edit mapping &": r"Manual-edit mapping & $-$0.025 & $-$0.011 & 206.5 \\", "Prior altogether &": r"Prior altogether & $-$0.021 & $-$0.003 & 198.2 \\",
             "Retake features &": r"Retake features & $-$0.021 & $-$0.016 & 213.1 \\", "Style prior (generic kept)": r"Style prior (generic kept) & $-$0.015 & $-$0.008 & 194.7 \\", "Semantic features &": r"Semantic features & $-$0.003 & $-$0.003 & 190.3 \\",
             "Implicit feedback &": r"Implicit feedback & +0.004 & +0.002 & 177.9 \\", "History features &": r"History features & +0.005 & $-$0.005 & 186.2 \\", "Discounting &": r"Discounting & +0.003 & +0.018 & 168.5 \\", "Exploration &": r"Exploration & +0.027 & +0.026 & 140.8 \\"}.items():
    line(k, v)
line("Acceptance during training counts every", "Acceptance during training counts every exploratory proposal as a mistake, so it cannot say whether exploration led to a better policy. We therefore trained each variant on ten videos under three feedback conditions, froze its posterior mean, and scored it on four unseen videos with no corrections. Table~\\ref{tab:explore} gives the outcome over 48 runs per cell. Held-out acceptance barely depends on the exploration strategy: every paired difference from Thompson sampling at $v=0.25$ lies within about one point, and most intervals include zero. A purely greedy learner ends at 0.931 against 0.932 under normal feedback. The training cost, on the other hand, is large and visible to the creator, with 0.908 for greedy against 0.883 for $v=0.25$ and 0.851 for $v=0.5$.")
for k, v in {"Greedy & 0.908": r"Greedy & 0.908 & 0.931 & 0.884 & 0.917 & 0.867 & 0.908 \\", "Thompson, $v=0.05$ & 0.896": r"Thompson, $v=0.05$ & 0.897 & 0.934 & 0.873 & 0.914 & 0.852 & 0.902 \\", "Thompson, $v=0.25$ & 0.882": r"Thompson, $v=0.25$ & 0.883 & 0.932 & 0.855 & 0.921 & 0.837 & 0.900 \\",
             "Thompson, $v=0.5$ & 0.851": r"Thompson, $v=0.5$ & 0.851 & 0.933 & 0.810 & 0.918 & 0.800 & 0.906 \\", "LinUCB & 0.880": r"LinUCB & 0.880 & 0.936 & 0.841 & 0.921 & 0.825 & 0.911 \\", "$\\epsilon$-greedy, $\\epsilon=0.1$ & 0.859": r"$\epsilon$-greedy, $\epsilon=0.1$ & 0.860 & 0.932 & 0.832 & 0.913 & 0.816 & 0.903 \\"}.items():
    line(k, v)
line("Fig.~\\ref{fig:sensitivity} varies one hyperparameter", "Fig.~\\ref{fig:sensitivity} varies one hyperparameter at a time over six seeds per persona. Prior strength $\\lambda$ has a broad optimum near 2 to 10 late in training. Weak priors ($\\lambda=0.5$) lose the cold-start benefit, at 0.818 early, and very strong ones ($\\lambda=30$) learn too slowly (0.807 early, 0.884 late). That loss is what the product accepts for keep/cut alone, to prevent one click from causing large changes. Posterior scale $v$ has the largest effect of any parameter: at $v=2$ the agent is little better than a rule (0.697 late), and it improves monotonically down to the smallest value tried ($v=0.05$, 0.916 late). The exploration floor $\\epsilon$ behaves in the same way, from 0.913 at 0 to 0.819 at 0.2. Discount $\\gamma$ gives 0.918 late at 1.0 and 0.858 at 0.95. The implicit reward has a small effect, with lower values marginally better (0.909 at 0, 0.893 at 1).")
line("Robustness asks what happens", "Robustness asks what happens when the environment gets worse. Fig.~\\ref{fig:robust} shows three such stresses. With the label noise raised from 0 to 30 percent, acceptance on videos 8 to 10 falls steadily from 0.920 to 0.749, staying well above Rule (about 0.59). A creator who reviews just 10 percent of decisions still leads the agent to 0.825, and 30 percent reaches 0.873, against 0.902 at full review. Reluctance to correct matters little: dropping the correction probability from 1.0 to 0.2 lowers acceptance from 0.910 to 0.889, because accepts still carry information. Video length also has little effect. With 12, 30, and 60 phrases per video (about 67, 160, and 315 decisions), acceptance is 0.898, 0.902, and 0.912, consistent with the idea that learning is measured in decisions rather than videos.")
line("Generalization asks about conditions", "Generalization asks about conditions the model and its prior were not built for. Fig.~\\ref{fig:unseen} tests creators who were deliberately placed outside the model family and outside the training data of the prior, scored only on keep/cut, pause, and filler decisions since these personas never zoom. With nonlinear rules, LinTS reaches 0.883, well above Rule (0.511) and the best fixed choice per type (0.489), because binned features can approximate interval rules. With noisy feedback (20 percent wrong labels, 30 percent correction rate), it reaches 0.810, below the best fixed arm at 0.835, and learning is slow and uneven, with an interval of $\\pm$0.085. With arbitrary rules that depend on a hash of the unit, it ends at 0.499, below the best fixed arm at 0.568. Here nothing can be learned, and the agent chases noise. These two failures are informative: under heavy noise or arbitrary preferences, a constant policy is the better choice, and the model is not winning by construction.")
sub("\\raggedright Creators outside the model family and the prior's training data. The method generalizes to nonlinear and noisy preferences but loses to a constant policy when preferences are arbitrary.", "\\raggedright Creators outside the model family and the prior's training data. The method generalizes to nonlinear preferences but falls behind a constant policy under heavy noise and arbitrary preferences.")
sub("The real clips cannot be redistributed without the speaker's consent, so the real-audio results can be repeated only with other footage.", "The real clips cannot be redistributed without the rights holder's consent, so the real-audio results can be repeated only with other footage. Repeating the analysis three times on the two real clips in fresh processes gave identical transcripts and cuts.")
sub("A complete rerun of the comparison, ablation, sensitivity, and robustness groups involves about 2500 simulated creators and takes under half an hour on the laptop described above.", "A complete rerun of the comparison, ablation, sensitivity, and robustness groups involves about 2500 simulated creators and takes under half an hour on the laptop described above, and the deep baselines take about half an hour more.")
sub("it plans over an episode with a learned reward.", "it plans over an episode with a learned reward, and in the rhythm agent its actions change the state it then acts in.")
sub("Actions affect later states only weakly, through history features that also proved to have little effect.", "In most decisions actions affect later states only weakly, through history features that also proved to have little effect, and where the state does matter, a sequential agent did not beat a bandit.")
sub("Hyperparameters were tuned on the same simulator used for evaluation, which favors the method. The default $v=0.25$ was selected for the quality of its confidences and not for peak acceptance, and lower values score higher on acceptance, so this choice does not inflate the headline figure. We cannot rule out that tuning affected other results.", "Hyperparameters were tuned on the same simulator used for evaluation, which favors the method. For the later results we separated this: rhythm settings and the deep baselines were tuned on seeds not used for reporting, and the product's bandit settings were confirmed on fresh seeds. The default $v=0.25$ was selected for the quality of its confidences and not for peak acceptance, so it does not inflate the headline figure. We cannot rule out that earlier tuning affected other results.")
line("The analysis has been checked on only two clips", "The analysis has been checked on only three clips, two real and one with a synthetic voice, against references prepared from the transcripts, with word-level comparison and no formal listening test. The crew rules were written by looking at one of these clips. Only English is supported; other languages get a warning, not a usable edit. Running Whisper in a fresh process removes a difference we could reproduce but not explain, so it is a workaround and not a diagnosis. Fillers with no audio dip stay in the edit. Disfluency recovery can miss repeats that the CTC model does not read clearly, and Whisper can still mishear words such as names.")
sub("So the sequential part of this work is a length budget, not a rich model of pacing.", "So the planner's sequential content is a length budget, and the rhythm agent, the richer sequential model, has only been tested against simulated creators whose preferences a bandit can also see.")
sub("Scalability is not a concern for the learner, but analysis takes about as long as the clip on a laptop CPU.", "Scalability is not a concern for the learner, but analysis takes one to two times the clip's length on a laptop CPU, and uploads are limited to five minutes.")
line("This paper treated talking-head editing", "This paper treated talking-head editing as online RL from one creator's normal editing behavior, a setting in which rule-based tools cannot adapt and supervised models lack data. We combined a linear Thompson sampling bandit with a tempered style posterior for the cold start, an uncertainty-ordered review queue, an exact posterior-sampling planner for length targets, a randomized value-iteration agent for visual pacing, and propensity logging for offline evaluation. In simulation, the method reaches 91.0 percent acceptance by the tenth video against 58.6 percent for a rule baseline, with 73 percent less regret. Learned policies are strongly creator-specific (0.936 against 0.468 across creators), review at a 30 percent budget gives 86.9 percent correct decisions against 63.3 percent in timeline order, and the planner reduces length error from 40.0 to 4.6 percent for 0.4 points of acceptance.")
line("The experiments also set limits", "The experiments also set limits on these claims. They establish that structured priors, queueing, and planning add measurable value in the simulator, and that personalization has a large effect there. They do not establish that random exploration is valuable in this domain, since corrections carry enough information for a nearly greedy policy to do well, or that value iteration improves on a bandit for pacing. A neural agent nearly catches the linear methods by the tenth video, so the linear choice rests on early performance and cost. Nothing is yet established about real creators.")
sub("Future work follows directly. A study with real creators,", "Future work follows directly. A study with real creators,")
sub("History features could return to the planner with uncertainty penalties \\cite{yu2020}, now that we know how they fail.", "History features could return to the planner with uncertainty penalties \\cite{yu2020}, now that we know how they fail, and RLSVI should be tested against creators who mostly give delayed feedback, where a bandit has less to learn from. Crew-chatter detection needs a labeled set and a learned model.")
sub("Claude, an AI model from Anthropic, was used to assist with software development and with drafting and editing this manuscript. The authors reviewed the content and take responsibility for it.", "Claude, an AI model from Anthropic, was used to assist with software development, experiments, and drafting and editing this manuscript. The authors reviewed the content and take responsibility for it.")

# ---------------------------------------------------------------- references
after(r"\bibitem{osband2013}", "\n\\bibitem{osband2016} I. Osband, B. Van Roy, and Z. Wen, ``Generalization and exploration via randomized value functions,'' in \\emph{Proc. 33rd Int. Conf. Mach. Learn.}, New York, NY, USA, 2016, pp. 2377--2386.\n\n\\bibitem{osband2019} I. Osband, B. Van Roy, D. J. Russo, and Z. Wen, ``Deep exploration via randomized value functions,'' \\emph{J. Mach. Learn. Res.}, vol. 20, no. 124, pp. 1--62, 2019.")
after(r"\bibitem{riquelme2018}", "\n\\bibitem{zhou2020neural} D. Zhou, L. Li, and Q. Gu, ``Neural contextual bandits with UCB-based exploration,'' in \\emph{Proc. 37th Int. Conf. Mach. Learn.}, 2020, pp. 11492--11502.")

# ---------------------------------------------------------------- length: 18-20 pages; these two figures repeat tables
def drop_figure(path: str) -> None:
    hits = [i for i, x in enumerate(lines) if x.startswith(r"\Figure") and path in x]
    assert len(hits) == 1, path
    del lines[hits[0]:hits[0] + 2]


drop("Three design choices matter for learning.")
drop_figure("figures/by_persona.pdf")
drop_figure("figures/calibration.pdf")
sub("Fig.~\\ref{fig:persona} and Table~\\ref{tab:persona} repeat", "Table~\\ref{tab:persona} repeats")
sub("(Fig.~\\ref{fig:calib}, Table~\\ref{tab:calib})", "(Table~\\ref{tab:calib})")

text = "\n".join(lines)
best = N["rhythm"]
text = (text.replace("@RLSVI_MS@", f"{NF['cost']['rlsvi_begin_video_ms_10_videos']:.0f}").replace("@NEURAL_S@", f"{ne['secs']:.0f}")
        .replace("@N_V10@", f"{ne['v10']:.3f}").replace("@N_DV10@", f"{-vp['v10'] * 100:.1f}").replace("@N_V1@", f"{ne['v1']:.3f}")
        .replace("@N_DR@", f"{vp['regret']:.0f}").replace("@NP_V1@", f"{np_['v1']:.3f}").replace("@NP_DR@", f"{vq['regret']:.0f}"))
dq, dw, pb, pw = best["DQN, blank"], best["DQN, warm start"], best["PPO, blank"], best["PPO, warm start"]
dbest = max(dq["final"], dw["final"])
pbest = max(pb["final"], pw["final"])
DT = (f"The deep agents had far less to work with, about 260 samples per run. The better DQN reached {dbest:.3f} and the better PPO {pbest:.3f}, "
      f"so at this data budget both trail the linear agents on matches even after tuning, although DQN's final rating is level with theirs and every interval on rating is wide.")
text = text.replace("@DEEP_TEXT@", DT)
assert "@" not in re.sub(r"[\w.]+@[\w.]+|e-mail", "", text.split(r"\begin{thebibliography}")[0]), [m for m in re.findall(r"@\w+@", text)]
(P / "main.tex").write_text(text, encoding="utf-8")
print("written", len(text))
