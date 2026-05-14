# Sound Morpher — Implementation Plan

**Date:** 2026-05-14  
**Stack:** Python 3.11+ · PySide6 · numpy · scipy · librosa · sounddevice · soundfile

---

## Fasering

De implementatie is opgedeeld in 9 fasen. Elke fase levert werkende, testbare software op — geen "big bang" aan het einde.

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                                        │
                                        ▼
Phase 9 ◄── Phase 8 ◄── Phase 7 ◄── Phase 5 ──► Phase 6
```

---

## Fase 1 — Project Foundation

**Doel:** werkende app-shell, donker thema, audio-engine basisklasse.

### Taken
- [ ] Mappenstructuur aanmaken (zie §Projectstructuur hieronder)
- [ ] `requirements.txt` met alle afhankelijkheden
- [ ] `main.py` — PySide6 `QApplication` + `MainWindow` (lege shell)
- [ ] Donker Qt-thema instellen (QPalette of QSS stylesheet)
- [ ] `AudioEngine` klasse — laad WAV, speel af via `sounddevice`
- [ ] CI-smoke test: app start en sluit zonder fout

### Deliverable
App opent een leeg venster met het juiste thema.

---

## Fase 2 — Audio Input

**Doel:** gebruiker kan geluid A en B laden via bestand of opname.

### Taken
- [ ] `SoundSlot` widget: waveform display, metadata (duur, sample rate), [▶] [●] knoppen
- [ ] Bestandskiezer: WAV selecteren → wordt gekopieerd naar project-intern geheugen
- [ ] Resample / bit-depth normalisatie bij laden (naar project-instellingen)
- [ ] `RecordingPanel` (modal): device-selector, VU-meter, Record/Stop, preview, assign-to A/B
- [ ] `sounddevice` input stream voor live opname
- [ ] Waveform rendering via `numpy` + `QPainter` of `pyqtgraph`

### Deliverable
Geluid A en B kunnen worden geladen of opgenomen; waveforms zijn zichtbaar.

---

## Fase 3 — Plugin Architectuur & Crossfade

**Doel:** morphing-pipeline actief met eerste plugin; stap-WAVs worden gegenereerd.

### Taken
- [ ] `MorphPlugin` abstracte basisklasse (zie REQUIREMENTS.md §7.1)
- [ ] `PluginRegistry`: scant `plugins/` map, importeert en registreert plugins
- [ ] Plugin-dropdown in UI (toont `plugin.name` + `plugin.description`)
- [ ] `CrossfadePlugin` — lineaire amplitude-interpolatie
- [ ] `MorphEngine.compute(plugin, audio_a, audio_b, steps, **params) → list[np.ndarray]`
- [ ] Berekening draait op `QThreadPool` worker (UI blokkeert niet)
- [ ] Voortgangsindicator tijdens berekening

### Deliverable
App genereert N crossfade-stappen; resultaat is hoorbaar via [▶] per stap.

---

## Fase 4 — Step Grid & Spectrogrammen

**Doel:** alle stappen zichtbaar als spectrogramtegel in scrollbare rij.

### Taken
- [ ] `StepTile` widget: stapnummer, spectrogramminiatuur, duur, actief-indicator
- [ ] `StepGrid` widget: horizontaal scrollbare `QScrollArea` met `StepTile`s
- [ ] STFT-berekening per stap (`scipy.signal.spectrogram` of `librosa.stft`)
- [ ] Spectrogramrendering naar `QPixmap` (kleurmap: viridis of custom)
- [ ] Asynchrone spectrogramberekening (na morph-compute, op achtergrondthread)
- [ ] Visuele onderscheiding van stap A (eerste) en stap B (laatste)
- [ ] Klik op tegel → speelt die stap af

### Deliverable
Volledige step grid zichtbaar met spectrogrammen; klikken speelt stap af.

---

## Fase 5 — Playback Engine

**Doel:** volledige sequentie-afspelen, inclusief BPM-gestuurd afspelen.

### Taken
- [ ] [▶ Play All] — speelt stap 1 t/m N sequentieel
- [ ] [■ Stop] — stopt huidige weergave
- [ ] `BpmEngine`: achtergrond-tick thread op basis van BPM + beats-per-step
- [ ] BPM-invoerveld (20–300), beats-per-step-selector, loop-toggle
- [ ] Tap-tempo knop (berekent BPM uit laatste 4 taps)
- [ ] Beat-indicator: visuele pulse (knoppen of LED-stijl indicator)
- [ ] Actieve-stap highlight in `StepGrid` synchroniseren met afspelen

### Deliverable
BPM-gestuurd afspelen werkt; beat-indicator pulseert; actieve tegel licht op.

---

## Fase 6 — Projectbestanden

**Doel:** sessies opslaan en herladen als `.smorph`.

### Taken
- [ ] `.smorph`-schrijver: ZIP met `project.json` + `audio/source_a.wav` + `audio/source_b.wav` + `audio/step_NN.wav`
- [ ] `.smorph`-lezer: laad project volledig terug (inclusief gecachede stap-WAVs)
- [ ] Menu: **File → New / Open / Save / Save As**
- [ ] Recent files (laatste 10, opgeslagen in `QSettings`)
- [ ] "Unsaved changes"-dialoog bij afsluiten of New
- [ ] `project.json` schema: plugin-naam, parameters, stappen, BPM, beats-per-step, sample rate, bit depth

### Deliverable
Project kan worden opgeslagen en volledig herladen inclusief alle stappen.

---

## Fase 7 — Overige Morph Plugins

**Doel:** drie extra plugins; plugin-parameterUI automatisch gegenereerd.

### Taken
- [ ] Plugin-parameter UI: `PluginParamPanel` rendert sliders/spinboxen/dropdowns op basis van `plugin.parameters`
- [ ] `SpectralFftPlugin` — FFT magnitude-interpolatie met phase vocoder
- [ ] `PitchShiftPlugin` — pitch van A verschuift naar gedetecteerde grondtoon van B (`librosa.effects.pitch_shift`)
- [ ] `GranularPlugin` — grains van A en B mengen per stap (`librosa` of eigen grain-engine)
- [ ] Parameters worden opgeslagen in `project.json`
- [ ] Wisselen van plugin triggert herberekening (met bevestigingsdialoog)

### Deliverable
Alle 4 plugins werken en zijn selecteerbaar; parameters worden opgeslagen.

---

## Fase 8 — Export

**Doel:** stappen exporteren als losse WAV-bestanden.

### Taken
- [ ] [↓ Export Steps] → OS-mapkiezer
- [ ] Exporteer `morph_step_01.wav` … `morph_step_NN.wav` in gekozen map
- [ ] Gebruik project-sample rate en bit depth
- [ ] `QProgressDialog` tijdens export
- [ ] Bevestigingsbericht na export met maplink

### Deliverable
Export werkt; bestanden zijn afspeelbaar in externe spelers.

---

## Fase 9 — Polish & Hardening

**Doel:** productiekwaliteit: foutafhandeling, prestaties, UX-verfijning.

### Taken
- [ ] Foutafhandeling uit REQUIREMENTS.md §14 implementeren (duurverschil, samplerate-mismatch, plugin-exception, opname-apparaat ontbreekt)
- [ ] Prestatietest: 8 stappen × 5s audio < 10s rekentijd (per plugin)
- [ ] Geheugengebruik idle < 200 MB controleren
- [ ] Toetsenbordnavigatie voor alle primaire acties
- [ ] Iconen toevoegen (toolbar, knoppen) via Qt-resources of SVG
- [ ] Installatiepackage bouwen met PyInstaller (Windows + macOS)
- [ ] Eindgebruikerstest op alle drie platforms

### Deliverable
Versie 1.0 release-kandidaat; installatiepackage beschikbaar.

---

## Projectstructuur

```
sound-morpher/
├── main.py                  # Entrypoint
├── requirements.txt
├── pyproject.toml           # Packaging metadata
├── app/
│   ├── __init__.py
│   ├── main_window.py       # MainWindow + layout
│   ├── audio_engine.py      # Laden, afspelen, opnemen
│   ├── morph_engine.py      # compute() pipeline + threading
│   ├── bpm_engine.py        # BPM tick thread
│   ├── project.py           # .smorph lezen/schrijven
│   └── widgets/
│       ├── sound_slot.py    # SoundSlot widget (waveform + controls)
│       ├── step_tile.py     # StepTile (spectrogram thumbnail)
│       ├── step_grid.py     # StepGrid (scrollbare rij)
│       ├── recording_panel.py
│       └── plugin_param_panel.py
├── plugins/
│   ├── base.py              # MorphPlugin abstracte klasse
│   ├── registry.py          # Plugin auto-discovery
│   ├── crossfade.py
│   ├── spectral_fft.py
│   ├── pitch_shift.py
│   └── granular.py
├── resources/
│   ├── theme.qss            # Dark theme stylesheet
│   └── icons/               # SVG iconen
└── tests/
    ├── test_audio_engine.py
    ├── test_morph_engine.py
    └── test_plugins.py
```

---

## Afhankelijkheden (`requirements.txt`)

```
PySide6>=6.6
numpy>=1.26
scipy>=1.12
librosa>=0.10
sounddevice>=0.4
soundfile>=0.12
pyqtgraph>=0.13        # waveform rendering (optioneel, anders QPainter)
```

---

## Volgorde van implementatie (samenvatting)

| Fase | Wat | Afhankelijk van |
|------|-----|-----------------|
| 1 | Foundation: app-shell, thema, audio-engine | — |
| 2 | Audio input: bestand + opname + waveform | 1 |
| 3 | Plugin-arch + Crossfade + compute-pipeline | 2 |
| 4 | Step grid + spectrogrammen | 3 |
| 5 | Playback engine (BPM) | 4 |
| 6 | Project files (.smorph) | 3, 5 |
| 7 | Overige plugins + param-UI | 3, 6 |
| 8 | Export | 6 |
| 9 | Polish, packaging | 1–8 |

---

*Plan opgesteld op 2026-05-14 op basis van REQUIREMENTS.md.*
