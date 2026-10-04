# LTM-AI — Latent Trackmania AI

> **Status: early design phase (v0).** No runnable agent yet. This repository currently holds the project's specification and design notes. The learning side of the project is still being worked out and several core questions remain open (see *Open questions*).
>
> **Language note:** this README is in English, but the specification and all other documents in this repository are written in **French** (I'm French).

---

## The idea

Drop an AI into a Trackmania 2020 track it has never seen, walk away, and come back to find it driving better than when you left.

To this day, no Trackmania AI has been able to beat human players on several traks without complete retraining. LTM-AI is an attempt to change that, by giving the agent a way to *adapt* to a new track on its own, in a reasonable amount of time, on consumer hardware.

## Why this project exists

Most game-playing agents are impressive until you change the level. They are trained for thousands of hours on a fixed set of tracks, and the moment you hand them something new, they fall apart or need to be retrained from scratch — with a human in the loop every step of the way.

Trackmania is a good place to attack this problem. Tracks are short, deterministic, and infinitely varied. A human player handles a brand-new track in a very recognisable way: drive it once slowly to *see* it, then grind one tricky section at a time until the whole run flows. Nobody restarts from zero every lap; they build an internal picture of the track and refine their driving against it.

LTM-AI is an attempt to give a machine that same loop — **look, understand, try, refine** — with as little human intervention as possible. The long-term question behind it is not "can an AI drive Trackmania?" (it can) but "can an AI *adapt* to a new environment by itself, in a reasonable amount of time, on consumer hardware?"

## How the agent is meant to think

Rather than reacting frame by frame to raw pixels, the agent works inside a compressed internal representation — a *latent space* — in which it keeps two separate pictures in mind:

- **What the car is doing.** Speed, orientation, drift, whether it is airborne: a compact summary of the vehicle's state, learned from screen captures and in-game telemetry.
- **What the track looks like.** A compact description of the environment around the car: the road ahead, the turn that is coming, the obstacle on the left.

Keeping these apart is deliberate. The car behaves the same way whatever the track; the track is the same whatever the car does. Separating them is what should let knowledge about driving carry over from one track to the next, while knowledge about a given track is learned fresh each time.

On top of these two pictures sits a *world model*: a learned simulator that, given the current situation and an action, predicts what the next instant will look like — without touching the game. This is the piece that lets the agent *imagine* before it *acts*.

## A run

**1. Driving.** The agent drives the track using what it already knows about driving in general. On a track it has never seen, this is cautious and clumsy.

**2. Adapting.** Here is the heart of the project. Sector by sector, the agent asks: *where exactly should I be aiming, and how should I get there?* It generates several candidate intentions, evaluates them — ideally in its own imagination, by running its world model forward — and keeps the ones that get through the sector faster. Then it moves to the next sector. After a full pass, the whole run is a little better. Then it starts again.

**3. Learning in the background.** While all this happens, every recorded attempt feeds back into the models. The agent's understanding of how cars move, and of what this track looks like, keeps getting sharper. New versions of the models are rolled out with safeguards so a bad update cannot wreck a good run.

**4. A human, optionally.** A supervision interface shows what the agent is doing, lets a human take the wheel to demonstrate a section, or pause and inspect. The ambition is that this becomes less and less necessary over time.

## What is already decided

- **Game:** Trackmania 2020, read through an [Openplanet](https://openplanet.dev/) plugin for telemetry and control.
- **Architecture:** three cooperating processes — a *control centre* (orchestration and human interface), an *inference* process that drives, and a *training* process that learns from recorded runs. They share data on disk and talk over a lightweight local protocol.
- **Hardware target:** a single mid-range gaming PC (think 8 GB GPU, 16 GB RAM). This constraint shapes everything: models stay small, training and driving must share the machine.
- **Stack:** Python / PyTorch; HDF5 for recorded runs; Openplanet (AngelScript) for the game-side plugin.
- **Continuity:** the agent carries its general driving knowledge from track to track, and keeps a per-track memory it can return to.

## Open questions

These are the questions the design still has to answer before serious implementation begins. They are listed here on purpose: a project like this lives or dies on them.

- **What exactly does the agent see?** Capture resolution, number of frames kept in memory, and how they are summarised are not fixed.
- **What counts as "better"?** Sector time alone is a very sparse signal. A continuous notion of progress along the scouted route is under study.
- **Can a sector be replayed from the same state?** Evaluating several candidates in the real game requires putting the car back into an exact prior state, which the game does not natively support. Imagination-based evaluation would sidestep this.
- **One model or many?** The current draft splits the system into several separately trained pieces. Joint training of encoder, dynamics and decision heads, as in recent world-model research, is being seriously considered instead.
- **What is the track model for?** Whether a dedicated predictive model of the track is needed, or whether a simpler *localiser* against the scouting run would do, is unresolved.

## Where the project stands

| Stage | State |
| --- | --- |
| Written specification (FR) | Mostly complete, being tightened |
| Openplanet plugin & telemetry | Not started |
| Data recording pipeline | Not started |
| First encoders & world model | Not started |
| Scouting / Driving / Adapting loop | Not started |
| Supervision interface | Not started |

## Influences

The project draws on the world-model line of research (Dreamer, TD-MPC and their successors), on model-predictive control with sampling-based planners (CEM, MPPI), and on the existing Trackmania reinforcement-learning community, whose work made it clear both what is possible and what is still hard.

## Repository contents

- `LTM-AI_Specification.md` — the full specification (**French**).
- Further design notes and diagrams will be added as they are written.

## License

To be defined before the first code release.
