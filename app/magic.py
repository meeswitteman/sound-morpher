"""Magic: pick the settings for a natural morph between two sounds by itself.

Three stages, each usable on its own:

1. analyse() measures what kind of sound each source is: where it starts and
   dies away, whether it has a pitch, how much of it is attack, how noisy it is.
2. make_plan() turns two profiles into the length handling (align the starts,
   stretch or pad, DTW) and a shortlist of plugins with parameters that suit
   the pair. The rules encode what each plugin is good at.
3. run_magic() renders every candidate as a short probe morph, scores the
   probes with score_morph(), and renders the winner at the full step count.

The score is objective, so "best" means something specific: in a timbre space
(the cepstral envelope of the long-term log-mel spectrum, plus a coarse
loudness envelope), a natural morph walks a straight line from A to B in even
strides. A detour means the steps in between sound like neither source (a
vocoder's buzz, a smeared attack); uneven strides mean the morph lurches;
extra spectral flatness over the endpoints means added noise. For two pitched
sounds each step should also carry one clear pitch on the glide from A's to
B's, not two at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.morph_engine import prepare_sources, render_steps, with_original_endpoints

# Frames for every analysis here.
_N_FFT = 2048
_HOP = 512

# A frame counts as part of the sound when it is no more than this far under
# the loudest frame.
_ACTIVE_DB = -45.0

# At most this much of each sound is pitch-tracked; the character of a sample
# shows long before then.
_PITCH_SECONDS = 8.0

# Cepstral coefficients kept for the timbre envelope.
_N_CEPS = 20

# Periodicity from which a frame counts as having a pitch.
_VOICED = 0.6

# Seconds of each step that are pitch-tracked when scoring a pitched pair.
_CLARITY_SECONDS = 3.0

# Steps rendered per candidate while comparing. A, three steps between, B.
_PROBE_STEPS = 5

# Candidates are compared on this much of the sound. How a method treats a
# pair shows within seconds, and a half-minute bell would otherwise take
# minutes per candidate. The winner is rendered in full.
_PROBE_SECONDS = 6.0

# Length ratios. Under the first the lengths count as equal; above the second
# a phase-vocoder stretch would be heard as processing, so the shorter sound
# is padded instead.
_EQUAL_RATIO = 1.08
_MAX_STRETCH = 3.0

# Starts closer than this already line up.
_ALIGN_TOLERANCE_S = 0.010

# Small score handicap per place on the shortlist, so that when two candidates
# measure alike the one the analysis prefers wins.
_RANK_PENALTY = 0.03

# See Candidate.handicap.
_CROSSFADE_HANDICAP = 0.25


class MagicCancelled(Exception):
    """Raised inside run_magic() when the caller asked it to stop."""


# ── 1. Analysis ───────────────────────────────────────────────────────────────

@dataclass
class SoundProfile:
    duration: float      # seconds, the whole sample
    lead: float          # seconds of silence before the sound starts
    end: float           # seconds from the start of the sample to where it dies away
    voiced: float        # 0..1, share of the sounding frames that have a pitch
    f0: float            # median fundamental in Hz, 0 when unpitched
    f0_low: float        # 10th and 90th percentile of the fundamental
    f0_high: float
    f0_spread: float     # semitones between those two: vibrato, glides, melody
    percussive: float    # 0..1, share of the energy in the transient layer
    noisiness: float     # median spectral flatness, 0 = pure tone, 1 = white noise
    onsets: int          # distinct attacks: jumps in level, not wiggles in the spectrum
    hit: bool            # one sharp attack that then decays away: a one-shot

    @property
    def active(self) -> float:
        """Seconds between the start of the sound and where it dies away."""
        return max(self.end - self.lead, 0.0)

    @property
    def kind(self) -> str:
        # Stationary noise splits about evenly between HPSS's two layers, so
        # the transient share alone cannot tell a drum from a hiss; the
        # envelope can.
        if self.hit or (self.onsets >= 3 and self.percussive >= 0.55):
            return "percussive"
        if self.voiced >= 0.5:
            return "pitched"
        if self.noisiness >= 0.1 or self.percussive >= 0.35:
            return "noisy"
        return "tonal"

    def describe(self) -> str:
        parts = [f"{self.kind}", f"{self.active:.2f}s of sound"]
        if self.f0 > 0 and self.voiced >= 0.3:
            parts.append(f"pitch ~{self.f0:.0f} Hz")
        parts.append(f"{self.onsets} onset{'s' if self.onsets != 1 else ''}")
        return ", ".join(parts)


def _mono(audio: np.ndarray) -> np.ndarray:
    arr = audio.mean(axis=1) if audio.ndim == 2 else audio
    return np.asarray(arr, dtype=np.float32)


def _frame_db(mono: np.ndarray) -> np.ndarray:
    """RMS per analysis frame in dB relative to the loudest frame."""
    import librosa

    rms = librosa.feature.rms(y=mono, frame_length=_N_FFT, hop_length=_HOP)[0]
    peak = float(rms.max(initial=0.0))
    if peak <= 0.0:
        return np.full(len(rms), -np.inf)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(rms / peak)


def _sound_span(mono: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Seconds where the sound starts and where it has died away.

    Measured on a 10 ms envelope: the analysis frames are 46 ms long, and
    lining up two starts needs better than that.
    """
    from scipy.ndimage import uniform_filter1d

    win = max(1, int(0.010 * sample_rate))
    power = uniform_filter1d(mono.astype(np.float64) ** 2, win, mode="constant")
    env = np.sqrt(np.maximum(power, 0.0))   # the running mean can dip a hair under 0
    above = np.flatnonzero(env > env.max() * 10.0 ** (_ACTIVE_DB / 20.0))
    if not len(above):
        return 0.0, len(mono) / sample_rate
    start = max(above[0] - win // 2, 0)
    stop = min(above[-1] + win // 2, len(mono))
    return start / sample_rate, stop / sample_rate


def _count_onsets(db: np.ndarray, sample_rate: int) -> int:
    """Attacks: the level jumps 9 dB or more above its recent minimum.

    A spectral-flux onset detector also fires on every change of timbre inside
    a held note, and dozens of times a second on noise; this counts only what a
    listener would call a new event.
    """
    if len(db) < 2:
        return 0
    floor = np.where(np.isfinite(db), db, -120.0)
    back = max(1, int(0.15 * sample_rate / _HOP))
    # Lowest level over the frames just before each frame; a sound that starts
    # straight away counts as one attack.
    recent = np.array(
        [floor[max(0, i - back) : i].min() if i else -120.0 for i in range(len(floor))]
    )
    rising = (floor - recent >= 9.0) & (floor > -30.0)
    gap = max(1, int(0.08 * sample_rate / _HOP))
    count, last = 0, -gap
    for i in np.flatnonzero(rising):
        if i - last >= gap:
            count += 1
        last = i
    return count


def _is_hit(db: np.ndarray, sample_rate: int, onsets: int) -> bool:
    """One sharp attack, then a decay of 20 dB or more within a few seconds."""
    active = np.flatnonzero(db > _ACTIVE_DB)
    if onsets > 1 or len(active) < 2:
        return False
    start, stop = active[0], active[-1]
    peak = start + int(np.argmax(db[start : stop + 1]))
    attack = (peak - start) * _HOP / sample_rate
    duration = (stop - start) * _HOP / sample_rate
    tail = db[peak + (stop - peak) * 3 // 4 : stop + 1]
    # The frames are 46 ms long, so even an instant attack peaks a frame or
    # two in; a swell takes far longer.
    return bool(
        attack <= 0.07 and duration <= 4.0 and len(tail) and float(np.mean(tail)) <= -20.0
    )


def _periodicity(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per frame, how strongly the signal repeats at one pitch period, 0..1.

    The normalised autocorrelation peak over lags for 50-2000 Hz: near 1 for
    a clear pitch, near 0 for noise, and lower for two unrelated pitches at
    once. Only lags past the first zero crossing count; before it any
    low-passed signal still resembles itself, pitch or not. A small fraction
    of what pYIN costs, which matters when every probe step is measured.
    """
    import librosa

    padded = np.pad(mono.astype(np.float64), (_N_FFT // 2, _N_FFT // 2))
    if len(padded) < _N_FFT:
        return np.zeros(0)
    window = np.hanning(_N_FFT)
    frames = librosa.util.frame(padded, frame_length=_N_FFT, hop_length=_HOP) * window[:, None]
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, n=2 * _N_FFT, axis=0)) ** 2, axis=0)[:_N_FFT]
    # Undo the window's own taper, which would otherwise shrink long lags.
    w_ac = np.fft.irfft(np.abs(np.fft.rfft(window, n=2 * _N_FFT)) ** 2)[:_N_FFT]
    hi = min(int(sample_rate / 50.0), _N_FFT // 2)
    ac = ac[:hi] / np.maximum(w_ac[:hi, None], 1e-9 * w_ac[0])
    energy = ac[0]
    lo = max(1, int(sample_rate / 2000.0))
    lags = ac[lo:]
    negative = lags < 0
    crossed = np.cumsum(negative, axis=0) > 0
    peak = np.where(crossed, lags, -np.inf).max(axis=0)
    strength = np.where(np.isfinite(peak) & (energy > 0), peak / np.maximum(energy, 1e-20), 0.0)
    return np.clip(strength, 0.0, 1.0)


def analyse(audio: np.ndarray, sample_rate: int) -> SoundProfile:
    """Measure the properties of one sound that decide how to morph it."""
    import librosa

    mono = _mono(audio)
    duration = len(mono) / sample_rate
    if len(mono) < _N_FFT or not np.any(mono):
        return SoundProfile(
            duration, 0.0, duration, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False
        )

    db = _frame_db(mono)
    active = db > _ACTIVE_DB
    lead, end = _sound_span(mono, sample_rate)

    S = np.abs(librosa.stft(mono, n_fft=_N_FFT, hop_length=_HOP))
    harmonic, percussive = librosa.decompose.hpss(S)
    e_h = float(np.sum(harmonic ** 2))
    e_p = float(np.sum(percussive ** 2))
    perc_share = e_p / (e_h + e_p) if e_h + e_p > 0 else 0.0

    flatness = librosa.feature.spectral_flatness(S=S)[0]
    n = min(len(flatness), len(active))
    noisiness = float(np.median(flatness[:n][active[:n]])) if active[:n].any() else 0.0

    onsets = _count_onsets(db, sample_rate)
    hit = _is_hit(db, sample_rate, onsets)

    start = int(lead * sample_rate)
    segment = mono[start : start + int(_PITCH_SECONDS * sample_rate)]
    voiced, f0, f0_low, f0_high, spread = 0.0, 0.0, 0.0, 0.0, 0.0
    if len(segment) >= _N_FFT:
        strength = _periodicity(segment, sample_rate)
        track = librosa.yin(
            segment, fmin=50.0, fmax=2000.0, sr=sample_rate,
            frame_length=_N_FFT, hop_length=_HOP,
        )
        seg_active = _frame_db(segment) > _ACTIVE_DB + 10.0
        m = min(len(strength), len(track), len(seg_active))
        flag = strength[:m] >= _VOICED
        seg_active, track = seg_active[:m], track[:m]
        if seg_active.any():
            voiced = float(np.mean(flag[seg_active]))
        pitched = track[flag & seg_active & np.isfinite(track)]
        if len(pitched):
            f0 = float(np.median(pitched))
            f0_low, f0_high = (float(v) for v in np.percentile(pitched, [10, 90]))
            spread = 12.0 * float(np.log2(f0_high / f0_low))

    return SoundProfile(
        duration=duration, lead=lead, end=end, voiced=voiced, f0=f0,
        f0_low=f0_low, f0_high=f0_high, f0_spread=spread,
        percussive=perc_share, noisiness=noisiness, onsets=onsets, hit=hit,
    )


# ── 2. Plan ───────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    plugin: str
    params: dict[str, Any]
    reason: str
    # Added to the score. A crossfade measures well, because nothing in it
    # is processed, but it plays both sounds at once instead of morphing one
    # into the other, which a long-term spectrum cannot hear. It should win
    # only when every real morph is clearly worse.
    handicap: float = 0.0


@dataclass
class MagicPlan:
    pair: str                     # "pitched", "percussive", "texture" or "mixed"
    trim_a: tuple[float, float]   # seconds cut from the start and kept, for A
    trim_b: tuple[float, float]
    stretch_to_fit: bool
    dtw: bool
    candidates: list[Candidate]
    notes: list[str] = field(default_factory=list)


def _pair_kind(pa: SoundProfile, pb: SoundProfile) -> str:
    kinds = {pa.kind, pb.kind}
    if kinds == {"pitched"}:
        return "pitched"
    if "percussive" in kinds:
        return "percussive"
    if kinds <= {"noisy", "tonal"} and "noisy" in kinds:
        return "texture"
    return "mixed"


def _candidates(pair: str, pa: SoundProfile, pb: SoundProfile) -> list[Candidate]:
    if pair == "pitched":
        low = min(p.f0_low for p in (pa, pb) if p.f0 > 0)
        high = max(p.f0_high for p in (pa, pb) if p.f0 > 0)
        moving = max(pa.f0_spread, pb.f0_spread) > 1.0
        return [
            Candidate(
                "Pitch Shift",
                {
                    "fmin": float(np.clip(low * 0.5, 20.0, 500.0)),
                    "fmax": float(np.clip(high * 2.0, 500.0, 20000.0)),
                    "tracking": "dynamic" if moving else "median",
                    "formants": "preserve",
                },
                "both are pitched: glide one pitch into the other, each keeping its resonances",
            ),
            Candidate(
                "WORLD Vocoder",
                {"f0_mode": "interpolate", "pitch_detector": "harvest",
                 "frame_ms": 5.0, "envelope": "log", "channels": "stereo"},
                "both are pitched: interpolate pitch, envelope and breathiness separately",
            ),
            Candidate(
                "Spectral FFT",
                {"fft_size": "2048", "overlap": 75, "magnitude": "log",
                 "phase": "shortest-arc", "transients": "preserve"},
                "spectral morph with fine frequency resolution for the partials",
            ),
        ]
    if pair == "percussive":
        return [
            Candidate(
                "Spectral FFT",
                {"fft_size": "1024", "overlap": 75, "magnitude": "log",
                 "phase": "shortest-arc", "transients": "preserve"},
                "attacks stay sharp: transients morph on their own short frames",
            ),
            Candidate(
                "Vocoder",
                {"lpc_order": 24, "frame_ms": 20, "channels": "stereo"},
                "spectral envelope morph; short frames follow the decay",
            ),
            Candidate(
                "Crossfade",
                {"curve": "equal-power"},
                "artefact-free fallback for hits that share little spectrum",
                handicap=_CROSSFADE_HANDICAP,
            ),
        ]
    if pair == "texture":
        return [
            Candidate(
                "Spectral FFT",
                {"fft_size": "2048", "overlap": 75, "magnitude": "log",
                 "phase": "shortest-arc", "transients": "preserve"},
                "noisy, sustained material: a log-magnitude spectral morph",
            ),
            Candidate(
                "Granular",
                {"grain_ms": 120.0, "overlap": 0.5, "mode": "scatter",
                 "jitter_ms": 30.0, "pitch_jitter": 0.0, "seed": 0},
                "textures dissolve into each other grain by grain",
            ),
            Candidate(
                "Griffin-Lim",
                {"fft_size": "2048", "n_iter": 32, "magnitude": "log",
                 "seed_phase": True, "transients": "preserve"},
                "smooth spectral morph with re-estimated phase",
            ),
        ]
    return [
        Candidate(
            "Spectral FFT",
            {"fft_size": "2048", "overlap": 75, "magnitude": "log",
             "phase": "shortest-arc", "transients": "preserve"},
            "different kinds of sound: a spectral morph handles anything",
        ),
        Candidate(
            "Vocoder",
            {"lpc_order": 24, "frame_ms": 30, "channels": "stereo"},
            "moves A's spectral envelope into B's",
        ),
        Candidate(
            "Crossfade",
            {"curve": "equal-power"},
            "artefact-free fallback: the real morphs all came out worse",
            handicap=_CROSSFADE_HANDICAP,
        ),
    ]


def make_plan(pa: SoundProfile, pb: SoundProfile) -> MagicPlan:
    """Decide the length handling and the plugins worth trying for this pair."""
    notes: list[str] = []
    pair = _pair_kind(pa, pb)
    notes.append(f"A: {pa.describe()}")
    notes.append(f"B: {pb.describe()}")

    # Line the starts up: cut the later-starting sound's extra lead silence,
    # so the steps in between do not carry two attacks a moment apart.
    common_lead = min(pa.lead, pb.lead)
    cut_a = pa.lead - common_lead if pa.lead - common_lead > _ALIGN_TOLERANCE_S else 0.0
    cut_b = pb.lead - common_lead if pb.lead - common_lead > _ALIGN_TOLERANCE_S else 0.0
    if cut_a or cut_b:
        who, cut = ("A", cut_a) if cut_a else ("B", cut_b)
        notes.append(f"Aligned the starts: {cut * 1000:.0f} ms of silence before {who} skipped")

    len_a = max(pa.end - cut_a, 1e-3)
    len_b = max(pb.end - cut_b, 1e-3)
    ratio = max(len_a, len_b) / min(len_a, len_b)

    dtw = pa.onsets >= 3 and pb.onsets >= 3 and ratio <= _MAX_STRETCH
    stretch = False
    if dtw:
        notes.append(
            "Both have several events: DTW lines up matching moments before morphing"
        )
    elif ratio < _EQUAL_RATIO:
        notes.append("Lengths match: no stretching needed")
    elif pa.kind == "percussive" and pb.kind == "percussive":
        notes.append("Both are hits: padded rather than stretched, so the decays stay natural")
    elif ratio <= _MAX_STRETCH:
        stretch = True
        notes.append(f"Lengths differ {ratio:.1f}x: the shorter sound is stretched to fit")
    else:
        notes.append(
            f"Lengths differ {ratio:.1f}x, too far to stretch cleanly: padded instead"
        )

    # When stretching, the stretch should cover the sound itself and not
    # whatever silence follows it, so both are cut where they die away.
    if stretch or dtw:
        trim_a = (cut_a, len_a + 0.05)
        trim_b = (cut_b, len_b + 0.05)
    else:
        trim_a = (cut_a, pa.duration)
        trim_b = (cut_b, pb.duration)

    return MagicPlan(
        pair=pair,
        trim_a=trim_a,
        trim_b=trim_b,
        stretch_to_fit=stretch,
        dtw=dtw,
        candidates=_candidates(pair, pa, pb),
        notes=notes,
    )


def _apply_trim(audio: np.ndarray, trim: tuple[float, float], sample_rate: int) -> np.ndarray:
    start = int(round(trim[0] * sample_rate))
    stop = start + int(round(trim[1] * sample_rate))
    out = audio[start : min(stop, len(audio))]
    return out if len(out) >= _N_FFT else audio[start:]


# ── 3. Scoring ────────────────────────────────────────────────────────────────

@dataclass
class MorphScore:
    total: float
    detour: float      # path length / straight distance - 1
    unevenness: float  # coefficient of variation of the step sizes
    noise: float       # extra flatness over the endpoints, per 10 dB
    blur: float = 0.0  # lost pitch clarity in the steps between, pitched pairs only
    pitch: float = 0.0 # distance from the pitch glide A → B, per octave, pitched pairs only

    def __str__(self) -> str:
        text = (
            f"score {self.total:.3f}  (detour {self.detour:.2f}, "
            f"uneven {self.unevenness:.2f}, noise {self.noise:.2f}"
        )
        if self.blur or self.pitch:
            text += f", blur {self.blur:.2f}, pitch {self.pitch:.2f}"
        return text + ")"


def _timbre_features(
    steps: list[np.ndarray], sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per step: a level-independent timbre vector, and median flatness in dB.

    The vector holds the long-term log-mel spectrum (where the energy sits in
    frequency) and a coarse loudness envelope (how it unfolds in time), each
    relative to the step's own maximum: level matching already handles
    loudness, so only the shape should count.
    """
    import librosa
    from scipy.fft import dct

    length = max(len(s) for s in steps)
    n_seg = 32
    feats, flats = [], []
    for step in steps:
        mono = np.zeros(max(length, _N_FFT), dtype=np.float32)
        m = _mono(step)
        mono[: len(m)] = m
        S = np.abs(librosa.stft(mono, n_fft=_N_FFT, hop_length=_HOP)) ** 2
        mel = librosa.feature.melspectrogram(S=S, sr=sample_rate, n_mels=64)

        spec = 10.0 * np.log10(mel.mean(axis=1) + 1e-12)
        spec = np.maximum(spec - spec.max(), -80.0)
        # Keep only the envelope (the low cepstral coefficients, as MFCCs do).
        # The harmonic comb itself would make a pitch glide, which moves every
        # partial through bands it was not in, measure as a detour, while a
        # crossfade of two pitches sat on the straight line.
        spec = dct(spec, norm="ortho")[1:_N_CEPS]

        frame_power = mel.sum(axis=0)
        bounds = np.linspace(0, len(frame_power), n_seg + 1).astype(int)
        seg = np.array([
            frame_power[lo:hi].mean() if hi > lo else 0.0
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ])
        env = 10.0 * np.log10(seg + 1e-12)
        env = np.maximum(env - env.max(), -60.0)

        feats.append(np.concatenate([spec / np.sqrt(64), 0.7 * env / np.sqrt(n_seg)]))

        flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
        loud = frame_power > frame_power.max() * 1e-4
        value = float(np.median(flat[loud])) if loud.any() else 1.0
        flats.append(10.0 * np.log10(max(value, 1e-6)))
    return np.array(feats), np.array(flats)


def _pitch_reading(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Clarity (mean periodicity of the sounding frames, 0..1) and median pitch.

    Two pitches sounding at once, the crossfade of a pitched morph, read as
    less clearly pitched, and often as their common sub-octave, rather than
    as one pitch on its way from A's to B's. Pitch is 0 when there is none.
    """
    import librosa

    mono = _mono(audio)
    db = _frame_db(mono)
    active = np.flatnonzero(db > _ACTIVE_DB)
    if not len(active):
        return 0.0, 0.0
    start = active[0] * _HOP
    segment = mono[start : start + int(_CLARITY_SECONDS * sample_rate)]
    if len(segment) < _N_FFT:
        return 0.0, 0.0
    strength = _periodicity(segment, sample_rate)
    track = librosa.yin(
        segment, fmin=50.0, fmax=2000.0, sr=sample_rate,
        frame_length=_N_FFT, hop_length=_HOP,
    )
    loud = _frame_db(segment) > _ACTIVE_DB + 10.0
    m = min(len(strength), len(track), len(loud))
    strength, track, loud = strength[:m], track[:m], loud[:m]
    if not loud.any():
        return 0.0, 0.0
    clarity = float(np.mean(strength[loud]))
    voiced = track[loud & (strength >= _VOICED) & np.isfinite(track)]
    return clarity, float(np.median(voiced)) if len(voiced) else 0.0


def score_morph(
    steps: list[np.ndarray], sample_rate: int, pitched: bool = False
) -> MorphScore:
    """How natural a morph is; lower is better. See the module docstring.

    With `pitched`, two more things count against the morph: a step that is
    less clearly pitched than its place between A and B would have it, and a
    step whose pitch is off the glide from A's pitch to B's, up to an octave.
    """
    feats, flats = _timbre_features(steps, sample_rate)
    d = np.linalg.norm(np.diff(feats, axis=0), axis=1)
    # Floor on the straight distance: two near-identical sources would
    # otherwise turn any wobble into an enormous detour.
    direct = max(float(np.linalg.norm(feats[-1] - feats[0])), 2.0)
    detour = float(d.sum()) / direct - 1.0
    unevenness = float(d.std() / d.mean()) if d.mean() > 0 else 0.0

    n = len(steps)
    t = np.arange(n) / (n - 1)
    expected = (1.0 - t) * flats[0] + t * flats[-1]
    excess = np.maximum(flats[1:-1] - expected[1:-1], 0.0)
    noise = float(excess.mean()) / 10.0 if len(excess) else 0.0

    blur = pitch = 0.0
    if pitched and n > 2:
        readings = [_pitch_reading(s, sample_rate) for s in steps]
        clarity = np.array([r[0] for r in readings])
        expected = (1.0 - t) * clarity[0] + t * clarity[-1]
        blur = float(np.mean(np.maximum(expected[1:-1] - clarity[1:-1], 0.0)))

        f0 = np.array([r[1] for r in readings])
        if f0[0] > 0 and f0[-1] > 0:
            # One pitch gliding from A's to B's in equal musical steps.
            target = np.exp((1.0 - t) * np.log(f0[0]) + t * np.log(f0[-1]))
            miss = np.where(
                f0[1:-1] > 0,
                np.minimum(np.abs(np.log2(np.maximum(f0[1:-1], 1e-9) / target[1:-1])), 1.0),
                1.0,
            )
            pitch = float(np.mean(miss))

    total = detour + 0.5 * unevenness + 0.5 * noise + 2.0 * blur + pitch
    return MorphScore(total, detour, unevenness, noise, blur, pitch)


# ── Run ───────────────────────────────────────────────────────────────────────

@dataclass
class MagicResult:
    steps: list[np.ndarray]
    plugin: str
    params: dict[str, Any]
    stretch_to_fit: bool
    dtw: bool
    report: list[str]


def run_magic(
    registry,
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    steps: int,
    sample_rate: int,
    progress_cb: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> MagicResult:
    """Analyse, plan, try every candidate, and render the best at `steps`."""

    def progress(value: float) -> None:
        if progress_cb is not None:
            progress_cb(int(np.clip(value, 0, 100)))

    def check() -> None:
        if should_stop is not None and should_stop():
            raise MagicCancelled()

    progress(0)
    pa = analyse(audio_a, sample_rate)
    progress(5)
    check()
    pb = analyse(audio_b, sample_rate)
    progress(10)
    check()

    plan = make_plan(pa, pb)
    candidates = [c for c in plan.candidates if c.plugin in registry]
    trimmed_a = _apply_trim(audio_a, plan.trim_a, sample_rate)
    trimmed_b = _apply_trim(audio_b, plan.trim_b, sample_rate)
    a, b = prepare_sources(
        trimmed_a, trimmed_b, sample_rate, stretch_to_fit=plan.stretch_to_fit, dtw=plan.dtw
    )
    progress(15)
    check()

    # The probes run on the opening seconds, and are scored against the same
    # seconds of the sources as they were before stretching or warping.
    cut = int(_PROBE_SECONDS * sample_rate)
    probe_a, probe_b = a[:cut], b[:cut]
    ref_a, ref_b = trimmed_a[:cut], trimmed_b[:cut]

    report = list(plan.notes)
    probe_steps = min(steps, _PROBE_STEPS)
    reuse_probe = probe_steps == steps and len(a) <= cut and len(b) <= cut
    span = 70.0 / max(len(candidates), 1)
    scored: list[tuple[float, int, Candidate]] = []
    probes: dict[int, list[np.ndarray]] = {}
    for rank, cand in enumerate(candidates):
        base = 15.0 + rank * span
        plugin = registry.get(cand.plugin)
        try:
            probe = render_steps(
                plugin, probe_a, probe_b, probe_steps, sample_rate, cand.params,
                progress_cb=lambda done, base=base: progress(base + span * done / probe_steps),
            )
        except Exception as exc:  # a plugin that cannot handle this pair drops out
            report.append(f"  {cand.plugin}: skipped ({exc})")
            check()
            continue
        check()
        score = score_morph(
            with_original_endpoints(probe, ref_a, ref_b),
            sample_rate,
            pitched=plan.pair == "pitched",
        )
        line = f"  {cand.plugin}: {score}"
        if cand.handicap:
            line += f"  + {cand.handicap:.2f} handicap"
        report.append(line)
        scored.append((score.total + cand.handicap + rank * _RANK_PENALTY, rank, cand))
        if reuse_probe:
            probes[rank] = probe

    if not scored:
        raise RuntimeError("None of the candidate algorithms could morph these sounds")

    _, best_rank, best = min(scored, key=lambda item: (item[0], item[1]))
    report.insert(len(plan.notes), f"Tried {len(scored)} algorithm(s) for a {plan.pair} pair:")
    report.append(f"Chosen: {best.plugin} — {best.reason}")

    if best_rank in probes:
        result = probes[best_rank]
    else:
        result = render_steps(
            registry.get(best.plugin), a, b, steps, sample_rate, best.params,
            progress_cb=lambda done: progress(85.0 + 15.0 * done / steps),
        )
    result = with_original_endpoints(result, audio_a, audio_b)
    progress(100)

    return MagicResult(
        steps=result,
        plugin=best.plugin,
        params=dict(best.params),
        stretch_to_fit=plan.stretch_to_fit,
        dtw=plan.dtw,
        report=report,
    )
