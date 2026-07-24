# GridSense . Demo Video Script

Target: about **2:45 to 3:00**. The hackathon allows up to 3 minutes, and
submissions that use most of that window tend to place better than the ones
that stop at 60 seconds, because there's simply more room to show the thing
working and make the case for why it matters. Judging is peer-voted on
"Overall Innovation," so open with something that makes someone stop
scrolling, then earn the rest of the time.

Record your screen with **OBS Studio**, Windows **Game Bar** (`Win + G`), or
Zoom's "share screen + record."

---

### Scene 1 . Cold open (0:00-0:12)
> *[Black screen, or `pitch_deck.html` slide 1 with the audio muted. Play a
> friction clip from `sample_data/friction/` first, before saying a word.
> Let the grinding sound run under silence for about 2 seconds.]*

**No dialogue yet.** Let the bad sound do the hook. Then:

> "That's a bearing about to fail. You just heard it happening. Most
> maintenance software never will, because it was never built to listen."

### Scene 2 . Introduce the problem, in human terms (0:12-0:35)
> *[`pitch_deck.html` slide 1, "Most of us just wait for things to break."]*

> "Here's the thing: every machine tells you when it's starting to fail.
> A dryer squeals before it stops spinning. A fan wobbles before it seizes.
> We just don't act on it, because the tools that catch this early were
> built for factories, not for the rest of us. Vibration sensors, cloud
> dashboards, thousands of dollars a machine. So small workshops, clinics,
> classrooms, and family farms get nothing. They run everything until it
> breaks."

### Scene 3 . The idea, in plain language (0:35-0:55)
> *[`pitch_deck.html` slide 2, the three-step "how it works"]*

> "I built GridSense to fix that with the one sensor almost everyone already
> owns: a microphone. You record a few seconds of a machine running, and it
> listens the way an experienced technician listens, breaking the sound
> apart into its frequencies and comparing it to what healthy and failing
> machines actually sound like. No new hardware, nothing to install. Let me
> show you it working, live."

### Scene 4 . Live demo: HEALTHY, then FRICTION, then IMBALANCE (0:55-1:55)
> *[Switch to the running Streamlit app, `python run.py`. Load a sample from
> each class in turn from the sidebar, or drag in `.wav` files directly.]*

> "First, a healthy motor. Clean, even hum. GridSense gives it a health
> score around 100, a green badge, nothing to do.
>
> Now here's a bearing with friction. Watch the spectrogram, that bright
> band lighting up high in the frequency range is the grind we heard at the
> very start of this video. The health score drops into the red, and it
> doesn't just flag a problem, it tells you the fix: inspect and
> re-lubricate the bearing.
>
> And this one is mechanical imbalance, a slow wobble in the waveform a few
> times a second, from a rotor spinning slightly off-center. GridSense
> recommends checking the rotor balance and the mounting bolts, the two
> cheapest things to rule out first."

*(If you have access to a real fan or motor, record one live demo from an
actual microphone here instead of a sample file. It's the single biggest
credibility boost available and takes thirty extra seconds.)*

### Scene 5 . Results, and the honest part (1:55-2:20)
> *[`pitch_deck.html` slide 4, the accuracy stat]*

> "On data the model had never seen before, GridSense got this right 98.3%
> of the time. I want to be upfront about the one time it didn't: it read a
> very early, very quiet friction case as healthy. That's the genuinely hard
> part of this problem: catching a fault in its quietest, earliest stage.
> I'd rather show you that limitation than hide it behind a clean-looking
> number."

### Scene 6 . Why it matters, and what's next (2:20-2:45)
> *[`pitch_deck.html` slides 5-7, "why it's different" and roadmap]*

> "Everything you just watched runs entirely on this laptop. No cloud, no
> account, no subscription, and nothing you record ever leaves the device.
> That matters, because it means this works anywhere there's a microphone:
> factory motors, HVAC systems, medical equipment, farm pumps, even a
> kitchen appliance. Next, I want to test it on real machinery in the field,
> teach it new failure sounds like electrical arcing and gear wear, and
> eventually build a tiny standalone version that listens on its own."

### Scene 7 . Close (2:45-2:58)
> *[`pitch_deck.html` slide 8, the close]*

> "GridSense: predictive maintenance for everyone, not just the Fortune
> 500. Thanks for watching."

---

## Recording tips
- **Play the cold-open sound before you say anything.** A silent two-second
  grind is worth more than any slide for stopping the scroll.
- Do one full dry run first. Pre-load the app and keep all three
  `sample_data` folders open in a file browser so you're not hunting for
  files on camera.
- Use `pitch_deck.html` for the slide beats (fullscreen it, arrow keys to
  advance) and cut to the real Streamlit app for the live demo. Judges
  respond well to seeing the polished pitch and the working product in the
  same video.
- Keep browser zoom high (`Ctrl`/`Cmd` + `+`) so the health score and badges
  read clearly on a phone screen, since a lot of peer voters watch on
  mobile.
- Add burned-in captions, or at least on-screen labels for each verdict
  (Healthy, Friction, Imbalance), since a large share of viewers watch
  muted while skimming a long list of submissions.
- Aim for the full length rather than cutting it short. A confident,
  unhurried 2:50 reads as more finished than a rushed 1:20, and it gives you
  room to actually show the product instead of just describing it.
- Export at 1080p, upload to **YouTube** (unlisted or public), and paste the
  link into the Devpost submission form well before the deadline in case
  processing takes a few minutes.
