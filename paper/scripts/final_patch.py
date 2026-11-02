"""Applies the final fair-comparison results (data/final_summary.json) to main.tex. Run once, on the main.tex of commit 4d16199."""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "main.tex"
s = P.read_text(encoding="utf-8").replace("\r\n", "\n")


def rep(a: str, b: str) -> None:
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


# abstract
start, end = s.index(r"\begin{abstract}") + len(r"\begin{abstract}") + 1, s.index(r"\end{abstract}")
s = s[:start] + ("Short talking-head videos are cut by hand, and two creators rarely want the same cut from the same footage. Fixed rules cannot capture that difference, and supervised models need a labeled history that a new creator lacks. We treat each edit as a decision in online reinforcement learning, with the creator's accepts, rejections, and ordinary transcript edits as reward. A linear Thompson sampling agent handles the per-phrase decisions. Around it we add a tempered posterior over editing styles for the cold start, a review queue ordered by posterior uncertainty, an exact posterior-sampling planner for target lengths, and a randomized value-iteration agent for visual pacing. Across six simulated creators and ten seeds, the method reached 93.1 percent acceptance by the tenth video, against 58.6 percent for a rule baseline, with 9 percent fewer wrong proposals than the best fairly tuned neural agent. Networks draw level on acceptance by the fiftieth video. The planner cut held-out length error from 40.0 to 4.6 percent for 0.4 points of acceptance. Value iteration trailed a bandit on beat accuracy but earned better ratings when per-beat feedback was scarce. Three recorded clips exercised the analysis pipeline; no formal human study has been run.\n") + s[end:]

# contributions and related work
rep("On visual pacing with a delayed rating, randomized value iteration ties a bandit on the same features, and both match the creator better than tuned DQN and PPO agents at this data budget.",
    "On visual pacing, randomized value iteration trails a bandit on per-beat accuracy but earns better whole-video ratings once per-beat feedback is scarce. Against fairly tuned networks, the linear method makes fewer mistakes over 50 videos, although the networks draw level on acceptance.")
rep("Our results agree with that finding at this data scale.", "Our results agree with that finding over the first videos, and they also show networks closing the gap as data accumulates.")

# hyperparameters
rep(r"Neural bandit: hidden, batch, lr, $\epsilon$ & \multicolumn{2}{c|}{64--64, 32, $10^{-3}$, 0.05} \\",
    r"""Retuned $\lambda$, $v$, $\epsilon$ (Section~\ref{sec:fair}) & \multicolumn{2}{c|}{2, 0.15, 0} \\
Neural: hidden, batch, lr, $\epsilon$, steps per event & \multicolumn{2}{c|}{64--64, 32, $3{\times}10^{-4}$, 0, 4} \\
NeuralLinear: lr, $v$, refit interval & \multicolumn{2}{c|}{$10^{-3}$, 0.1, 32} \\""")

# benchmark table
rep(r"Neural $\epsilon$-greedy, blank & 0.695 & 0.874 & 0.897 $\pm$ 0.011 & -- & 0.853 & 538.5 & 225.3 $\pm$ 14.3 & -- \\",
    r"Neural, blank & 0.735 & 0.894 & 0.915 $\pm$ 0.011 & -- & 0.880 & 402.5 & 186.7 $\pm$ 13.6 & -- \\")
rep(r"Neural $\epsilon$-greedy, pretrained & 0.727 & 0.867 & 0.896 $\pm$ 0.012 & -- & 0.810 & 480.0 & 217.8 $\pm$ 13.6 & -- \\",
    r"""Neural, pretrained & 0.789 & 0.900 & 0.921 $\pm$ 0.011 & -- & 0.947 & 341.5 & 167.8 $\pm$ 14.2 & -- \\
NeuralLinear, blank & 0.677 & 0.880 & 0.907 $\pm$ 0.013 & -- & 0.883 & 528.0 & 219.6 $\pm$ 15.0 & -- \\
NeuralLinear, pretrained & 0.724 & 0.880 & 0.916 $\pm$ 0.012 & -- & 0.857 & 481.0 & 202.3 $\pm$ 13.6 & -- \\""")
rep(r"\textbf{LinTS, style prior (proposed)} & \textbf{0.806} & 0.890 & \textbf{0.910 $\pm$ 0.012} & 0.807 & \textbf{0.903} & \textbf{344.0} & \textbf{181.5 $\pm$ 11.0} & 6.19 \\",
    r"""LinTS, style prior (proposed) & 0.806 & 0.890 & 0.910 $\pm$ 0.012 & 0.807 & 0.903 & 344.0 & 181.5 $\pm$ 11.0 & 6.19 \\
\textbf{Proposed, retuned} & \textbf{0.826} & \textbf{0.909} & \textbf{0.931 $\pm$ 0.008} & -- & \textbf{0.967} & \textbf{334.0} & \textbf{152.7 $\pm$ 10.6} & -- \\""")
rep("A neural bandit run took 20 s against 2.4 s for LinTS.", "A neural bandit run with one gradient step per event took 20 s against 2.4 s for LinTS.")

# real audio
rep(" There are certainly more such problems in footage we have not tried. Nobody has yet reviewed the rendered audio by ear.",
    " There are certainly more such problems in footage we have not tried. We also ran the editor ourselves on these recordings during development and turned what we saw into fixes; this feedback was informal and is not a user study.")

# fair comparison
a = s.index(r"\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)[width=\textwidth]{figures/deep.pdf}")
b = s.index(r"\subsubsection{Visual Rhythm: Is a Sequential Agent Needed?}")
s = s[:a] + r"""\subsubsection{A Fair Test Against Neural Agents}
\label{sec:fair}
Our first deep baseline made the linear model look better than it deserved. That search tried four settings, and its best $\epsilon$ was the smallest on offer, a sign that the grid was too narrow. We repeated the comparison with one procedure for every family: settings chosen on seeds 201--203 by acceptance on videos 5 and 6, and results reported on the usual seeds, with each run extended to 50 videos. The neural agent keeps one network per decision type, trained online from a replay buffer, which is the DQN recipe \cite{mnih2015} with $\gamma=0$ because these decisions have no transition. Its grid now has 36 settings of learning rate, $\epsilon$, gradient steps per event, and width. We added NeuralLinear \cite{riquelme2018}, which fits a Bayesian linear model on the last hidden layer and acts by Thompson sampling. Pretrained variants learn first from the other creators' labels, the same information the generic prior has. Our own method went through 60 settings of $\lambda$, $v$, and $\epsilon$.

\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)[width=\textwidth]{figures/long.pdf}
{\parbox[t]{0.91\textwidth}{\raggedright Comparison over 50 videos with every family tuned on the same separate seeds. The retuned method leads early and keeps the lowest cumulative regret, while network-based agents draw level on acceptance.}\label{fig:deep}}

\begin{table}[t]
\caption{Fifty Videos, Six Creators, Ten Seeds}
\label{tab:long}
\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{|l|c|c|c|}
\hline
Method & Video 50 & Regret, 10 & Regret, 50 \\
\hline
LinTS, generic prior & 0.910 & 194.7 & 765.7 \\
Proposed, default & 0.915 & 181.5 & 713.3 \\
\textbf{Proposed, retuned} & 0.927 & \textbf{152.7} & \textbf{590.9} \\
Neural, blank & 0.929 & 186.7 & 642.9 \\
Neural, pretrained & 0.928 & 167.8 & 618.8 \\
NeuralLinear, blank & \textbf{0.933} & 219.6 & 668.9 \\
NeuralLinear, pretrained & \textbf{0.933} & 202.3 & 659.5 \\
\hline
\end{tabular}
\end{table}

The result changed our conclusion, as Fig.~\ref{fig:deep} and Table~\ref{tab:long} show. With wider tuning, the networks overtake our default settings: pretrained Neural makes 13.7 $\pm$ 6.9 fewer wrong proposals over ten videos and 94.5 $\pm$ 19.4 fewer over fifty. Retuning (Table~\ref{tab:hyper}) restores the linear method's lead. It makes 15.0 $\pm$ 6.7 fewer wrong proposals than pretrained Neural over ten videos ($p<0.001$) and 27.9 $\pm$ 19.7 fewer over fifty ($p=0.006$), and it starts higher, at 0.826 against 0.789. On 120 fresh runs (seeds 301--320), the retuned settings beat the defaults by 1.5 $\pm$ 0.5 points at video 10 and by 30.8 $\pm$ 3.6 wrong proposals. By video 50, however, the retuned method's acceptance is statistically level with Neural (0.927 against 0.929) and slightly below NeuralLinear (0.933, $p=0.09$).

Retuning has a price that the acceptance figures hide. Its confidences are less useful, with calibration error 0.052 against 0.023 and an area under the curve of 0.831 against 0.893, and only 6.0 percent of suggestions fall below the review threshold instead of 10.1. A designer who relies on the review queue may reasonably keep the default, which is why most results in this paper use it. Two further cautions apply. The chosen neural learning rate was the lowest in its grid, so a still lower one might do better, and all of these creators can be described by our features, which favors a linear model. The case for the linear method therefore rests on fewer early mistakes, cost, exact replay, and readable explanations. It is not a claim that networks are worse, and with enough feedback a network is a reasonable choice.

""" + s[b:]

# rhythm
rep(" A fair test of RLSVI needs creators who give mostly delayed feedback, and we have not run one.", r""" These creators give feedback on every beat, which leaves little for a delayed reward to add.

To give RLSVI its fair test, we reran the agents for 16 videos while withholding per-beat feedback, down to the whole-video rating alone (Table~\ref{tab:delayed}). Here the picture splits. RLSVI's rating overtakes the bandit's as feedback thins, by 0.21 $\pm$ 0.11 at 10 percent and by 0.48 $\pm$ 0.15 at rating only (both $p<0.001$), while its beat match falls well behind (0.336 against 0.515). In other words, it learns how often a creator wants beats without learning where to put them. Part of this rating gap is built in, because a bandit in this setup cannot use the rating at all. DQN receives the rating too and does not benefit (rating $-$0.31). All settings were tuned under full feedback, so these figures may understate both deep and randomized agents.

\begin{table}[t]
\caption{Visual Rhythm With Sparse Feedback (Video 16, Match / Rating)}
\label{tab:delayed}
\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{|l|c|c|c|}
\hline
Beats with feedback & RLSVI & Bandit & DQN, warm \\
\hline
100\% & 0.915 / +0.06 & 0.945 / +0.03 & 0.906 / +0.00 \\
30\% & 0.825 / +0.18 & 0.910 / +0.06 & 0.808 / +0.04 \\
10\% & 0.687 / +0.24 & 0.828 / +0.04 & 0.670 / +0.01 \\
Rating only & 0.336 / +0.16 & 0.515 / $-$0.32 & 0.276 / $-$0.31 \\
\hline
\end{tabular}
\end{table}""")

# generalization
a = s.index("With nonlinear rules, LinTS reaches 0.883")
b = s.index(r"\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)[width=\textwidth]{figures/unseen.pdf}")
s = s[:a] + r"""By video 8 (Fig.~\ref{fig:unseen}), LinTS reaches 0.883 with nonlinear rules, well above Rule (0.511) and the best fixed choice per type (0.489), because binned features can approximate interval rules. With noisy feedback (20 percent wrong labels, 30 percent correction rate) it reaches 0.810, still below the best fixed arm at 0.835. With arbitrary rules that depend on a hash of the unit, it ends at 0.499, below the fixed arm at 0.568, because nothing can be learned there and the agent chases noise.

Table~\ref{tab:outside} extends these creators to 30 videos and adds the tuned networks, with eight seeds per cell, so intervals are wide. Each family wins where its assumptions hold. On nonlinear rules the network makes roughly half as many mistakes as LinTS. Under noise the order reverses: LinTS passes the fixed arm and beats blank Neural by 7.3 $\pm$ 3.8 points. On arbitrary rules every learner stays below a constant policy, so the method is not winning by construction.

\begin{table}[t]
\caption{Creators Outside the Model Family, 30 Videos (Acceptance / Wrong Proposals)}
\label{tab:outside}
\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
Creator & LinTS & Neural & Neural, pretr. & Fixed \\
\hline
Nonlinear & 0.931 / 454 & 0.948 / 244 & 0.953 / 230 & 0.479 \\
Noisy & 0.953 / 541 & 0.880 / 564 & 0.873 / 532 & 0.836 \\
Arbitrary & 0.508 / 1159 & 0.511 / 1140 & 0.477 / 1142 & 0.559 \\
\hline
\multicolumn{5}{p{80mm}}{Acceptance counts keep/cut, pause, and filler decisions; wrong proposals count all decisions.}
\end{tabular}
\end{table}

""" + s[b:]
rep("The method generalizes to nonlinear preferences but falls behind a constant policy under heavy noise and arbitrary preferences.", "The first eight videos. The method generalizes to nonlinear preferences but is still behind a constant policy under heavy noise and arbitrary preferences.")

# reproducibility, RL claim, threats, limits
rep("and the deep baselines take about half an hour more.", "and the final deep comparison takes about an hour and a half.")
rep("and where the state does matter, a sequential agent did not beat a bandit.", "and where the state does matter, a sequential agent beat a bandit only on whole-video rating under scarce feedback.")
rep("rhythm settings and the deep baselines were tuned on seeds not used for reporting, and the product's bandit settings were confirmed on fresh seeds.", "rhythm settings, the deep baselines, and the retuned linear settings were tuned on seeds not used for reporting, and the retuned and product settings were confirmed on fresh seeds.")
rep("The most serious limitation is that no human study has been run.", "The most serious limitation is that no formal human study has been run.")

# conclusion
a = s.index("This paper treated talking-head editing as online RL")
b = s.index("Future work follows directly.")
s = s[:a] + r"""This paper treated talking-head editing as online RL from one creator's normal editing behavior, a setting in which rule-based tools cannot adapt and supervised models lack data. We combined a linear Thompson sampling bandit with a tempered style posterior for the cold start, an uncertainty-ordered review queue, an exact posterior-sampling planner for length targets, a randomized value-iteration agent for visual pacing, and propensity logging for offline evaluation. In simulation, retuned settings reach 93.1 percent acceptance by the tenth video against 58.6 percent for a rule baseline, and they make the fewest wrong proposals over 50 videos of any agent we tried. Learned policies are strongly creator-specific (0.936 against 0.468 across creators), review at a 30 percent budget gives 86.9 percent correct decisions against 63.3 percent in timeline order, and the planner reduces length error from 40.0 to 4.6 percent for 0.4 points of acceptance.

The experiments also set limits on these claims. Fairly tuned networks overtook our default settings and drew level with the retuned ones on acceptance by the fiftieth video, and they did better on nonlinear taste, so the linear choice rests on scarce-feedback performance, cost, and transparency. Retuning for fewer mistakes weakened the confidence signal that the review queue needs. Random exploration did not prove valuable here, because corrections carry enough information for a nearly greedy policy to do well, and value iteration improved on a bandit only for whole-video rating under scarce feedback. Nothing is yet established about real creators.

""" + s[b:]
rep("and RLSVI should be tested against creators who mostly give delayed feedback, where a bandit has less to learn from.", "and RLSVI should be tuned for sparse feedback so that its rating gain can be tested with people.")
P.write_text(s, encoding="utf-8")
abstract = s[s.index(r"\begin{abstract}") + 16:s.index(r"\end{abstract}")]
print("abstract words", len(abstract.split()))
