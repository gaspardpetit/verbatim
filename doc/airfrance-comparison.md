# Air France Comparison

This document compares multiple transcription paths on the same bilingual Air France safety sample:

- Audio: `ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav`
- Languages: English + French
- Diarization: `pyannote`

The goal is to compare backend quality and the effect of Verbatim's code-switching-aware pipeline on the same source material.

## Systems

| System | Backend | Code-switch aware | Language detection | Diarization | Notes |
|---|---|---:|---|---|---|
| Whisper-3-Large | faster-whisper | No | MMS / single-pass | pyannote | Baseline |
| Whisper-3-Large-Verbatim | faster-whisper | Yes | MMS | pyannote | Iterative confirmation |
| Qwen3-ASR | Qwen3-ASR | No | MMS / single-pass | pyannote | Baseline |
| Qwen3-ASR-Verbatim | Qwen3-ASR | Yes | MMS | pyannote | Iterative confirmation |
| VibeVoice-ASR | VibeVoice-ASR | Unknown | External | Built-in / external | External baseline |

## Repro Commands

### Whisper-3-Large

```powershell
.\.venv\Scripts\python.exe -m verbatim `
  "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav" `
  -vvv `
  --diarize pyannote `
  --languages en fr `
  --code-switching false `
  --txt `
  --json `
  --outdir out\airfrance_whisper_mms_naive `
  --workdir out\airfrance_whisper_mms_naive_work `
  --transcriber-backend auto `
  --language-identifier-backend mms `
  --mms-lid-model-size facebook/mms-lid-126 `
  --log-file out\airfrance_whisper_mms_naive_clean.log
```

### Whisper-3-Large-Verbatim

```powershell
.\.venv\Scripts\python.exe -m verbatim `
  "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav" `
  -vvv `
  --diarize pyannote `
  --languages en fr `
  --txt `
  --json `
  --outdir out\airfrance_whisper_mms `
  --workdir out\airfrance_whisper_mms_work `
  --transcriber-backend auto `
  --language-identifier-backend mms `
  --mms-lid-model-size facebook/mms-lid-126 `
  --log-file out\airfrance_whisper_mms_clean.log
```

### Qwen3-ASR

```powershell
.\.venv\Scripts\python.exe -m verbatim `
  "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav" `
  -vvv `
  --diarize pyannote `
  --languages en fr `
  --code-switching false `
  --txt `
  --json `
  --outdir out\airfrance_qwen_mms_naive `
  --workdir out\airfrance_qwen_mms_naive_work `
  --transcriber-backend qwen `
  --language-identifier-backend mms `
  --mms-lid-model-size facebook/mms-lid-126 `
  --log-file out\airfrance_qwen_mms_naive_clean.log
```

### Qwen3-ASR-Verbatim

```powershell
.\.venv\Scripts\python.exe -m verbatim `
  "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav" `
  -vvv `
  --diarize pyannote `
  --languages en fr `
  --txt `
  --json `
  --outdir out\airfrance_qwen_mms `
  --workdir out\airfrance_qwen_mms_work `
  --transcriber-backend qwen `
  --language-identifier-backend mms `
  --mms-lid-model-size facebook/mms-lid-126 `
  --log-file out\airfrance_qwen_mms_clean.log
```

### VibeVoice-ASR

```text
External run. Current artifacts:
- out/airfrance_vibevoice_asr/vibevoice_asr.txt
- out/airfrance_vibevoice_asr/vibevoice_asr.json
- out/airfrance_vibevoice_asr/compare_summary.json
```

## Common Excerpt

The following excerpt is a useful stress case because it contains repeated safety instructions, language alternation, and closely related phrasing:

```text
Le gilet de sauvetage est situe sous votre siege ou dans l'accoudoir central.
Your life jacket is under your seat or in the central armrest.
Passez la tete dans l'encolure, attachez et serrez les sangles.
Place it over your head and pull the straps tightly around your waist.
```

## Output Comparison

### Whisper-3-Large

```text
Le gilet de sauvetage est situe sous votre siege ou dans la coudoir centrale.
Passez la tete dans l'encolure, attachez et serrez les sangles.
Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.
```

### Whisper-3-Large-Verbatim

```text
[SPEAKER_00][fr] Le gilet de sauvetage est situe sous votre siege ou dans la coudoir centrale.
[SPEAKER_01][en] Your life jacket is under your seat or in the central armrest.
[SPEAKER_00][fr] Passez la tete dans l'encolure, attachez et serrez les sangles.
[SPEAKER_01][en] Place it over your head and pull the straps tightly around your waist.
Inflate your life jacket by pulling the red toggles.
[SPEAKER_00][fr] Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.
[SPEAKER_01][en] Do this only when you are outside the aircraft.
```

### Qwen3-ASR

```text
[MISSING]
```

### Qwen3-ASR-Verbatim

```text
[SPEAKER_00][fr] Le gilet de sauvetage est situe sous votre siege ou dans l'accoudoir central.
[SPEAKER_01][en] Your life jacket is under your seat or in the central armrest.
[SPEAKER_00][fr] Passez la tete dans l'encolure, attachez et serrez les sangles.
[SPEAKER_01][en] Place it over your head and pull the straps tightly around your waist.
Inflate your life jacket by pulling the red toggles.
[SPEAKER_00][fr] Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.
[SPEAKER_01][en] Do this only when you are outside the aircraft.
```

### VibeVoice-ASR

```text
[SPEAKER_UNKNOWN][151.14-156.31] Le gilet de sauvetage est situe sous votre siege ou dans la coude-oreille centrale.
[SPEAKER_01][156.53-159.95] Your life jacket is under your seat or in the central armrest.
[SPEAKER_UNKNOWN][159.95-164.62] Placez la tete dans l'encadrement, attachez et serrez les sangles.
[SPEAKER_01][164.83-172.18] Place it over your head and pull the straps tightly around your waist. Inflate your life jacket by pulling the red toggles.
[SPEAKER_UNKNOWN][172.25-178.15] Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.
[SPEAKER_01][178.35-181.50] Do this only when you are outside the aircraft.
```

## Speaker-Turn Aligned Table

The table below aligns the current outputs by speaker-turn boundaries from the Verbatim runs. This keeps the systems row-aligned even when one run merges, drops, or mistranscribes a turn.

<table>
  <thead>
    <tr>
      <th>Anchor</th>
      <th>Whisper-3-Large</th>
      <th>Whisper-3-Large-Verbatim</th>
      <th>Qwen3-ASR</th>
      <th>Qwen3-ASR-Verbatim</th>
      <th>VibeVoice-ASR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>FR intro</code></td>
      <td>Madame, Monsieur, bonjour et bienvenue a bord.</td>
      <td>[SPEAKER_00][fr] Madame, Monsieur, bonjour et bienvenue a bord.</td>
      <td>[SPEAKER_00][fr] Madame, Monsieur, bonjour et bienvenue a bord.</td>
      <td>[SPEAKER_00][fr] Madame, Monsieur, bonjour et bienvenue a bord.</td>
      <td>[SPEAKER_UNKNOWN][0.00-3.02] Madame, <b>monsieur</b>, bonjour et bienvenue a bord.</td>
    </tr>
    <tr>
      <td><code>EN intro</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] Welcome aboard, ladies and gentlemen.<br>For your safety and comfort, please take a moment to watch the following safety video.</td>
      <td>[SPEAKER_01] Welcome aboard, ladies and gentlemen.<br>For your safety and comfort, please take a moment to watch the following safety video.</td>
      <td>[SPEAKER_01][en] Welcome aboard, ladies and gentlemen.<br>For your safety and comfort, please take a moment to watch the following safety video.</td>
      <td>[SPEAKER_01][3.02-10.88] Welcome aboard, ladies and gentlemen. For your safety and comfort, please take a moment to watch the following safety video.</td>
    </tr>
    <tr>
      <td><code>FR safety / seatbelt</code></td>
      <td>Ce film concerne votre securite a bord.<br>Merci de nous accorder votre attention.<br>Chaque fois que ce signal est allume, vous devez attacher votre ceinture pour votre securite.<br>Nous vous recommandons de la maintenir attachee de facon visible lorsque vous etes a votre siege.</td>
      <td>[SPEAKER_00][fr] Ce film concerne votre securite a bord.<br>Merci de nous accorder votre attention.<br>Chaque fois que ce signal est allume, vous devez attacher votre ceinture pour votre securite.<br>Nous vous recommandons de la maintenir attachee de facon visible lorsque vous etes a votre siege.</td>
      <td>[SPEAKER_00] Ce film concerne votre securite a bord.<br>Merci de nous accorder votre attention.<br>Chaque fois que ce signal est allume, vous devez attacher votre ceinture pour votre securite.<br>Nous vous recommandons de la maintenir attachee de facon visible lorsque vous etes a votre siege.</td>
      <td>[SPEAKER_00][fr] Ce film concerne votre securite a bord.<br>Merci de nous accorder votre attention.<br>Chaque fois que ce signal est allume, vous devez attacher votre ceinture pour votre securite.<br>Nous vous recommandons de la maintenir attachee de facon visible lorsque vous etes a votre siege.</td>
      <td>[SPEAKER_UNKNOWN][10.88-22.66] Ce film concerne votre securite a bord. Merci de nous accorder votre attention. Chaque fois que ce signal est allume, vous devez attacher votre ceinture pour votre securite.<br>[SPEAKER_UNKNOWN][23.12-29.09] Nous vous recommandons de la maintenir attachee de facon visible lorsque vous etes a votre siege.</td>
    </tr>
    <tr>
      <td><code>EN seatbelt</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] Whenever the seatbelt sign is on, your seatbelt must be securely fastened.<br>For your safety, we recommend that you keep your seatbelt fastened and visible at all times while seated.<br>To release the seatbelt, just lift the buckle.</td>
      <td>[SPEAKER_01] Whenever the seatbelt sign is on, your seatbelt must be securely fastened.<br>For your safety, we recommend that you keep your seatbelt fastened and visible at all times while seated.<br>To release the seatbelt, just lift the buckle.</td>
      <td>[SPEAKER_01][en] Whenever the seatbelt sign is on, your seatbelt must be securely fastened.<br>For your safety, we recommend that you keep your seatbelt fastened and visible at all times while seated.<br>To release the seatbelt, just lift the buckle.</td>
      <td>[SPEAKER_01][29.32-40.05] Whenever the seatbelt sign is on, your seatbelt must be securely fastened. For your safety, we recommend that you keep your seatbelt fastened and visible at all times while seated.<br>[SPEAKER_01][41.94-45.38] To release the seatbelt, just lift the buckle.</td>
    </tr>
    <tr>
      <td><code>FR smoking</code></td>
      <td>Pour detacher votre ceinture, soulevez la partie superieure de la boucle.<br>Il est strictement interdit de fumer dans l'avion, y compris dans les toilettes.</td>
      <td>[SPEAKER_00][fr] Pour detacher votre ceinture, soulevez la partie superieure de la boucle.<br>Il est strictement interdit de fumer dans l'avion, y compris dans les toilettes.</td>
      <td>[SPEAKER_00] Pour detacher votre ceinture, soulevez la partie superieure de la boucle.<br>Il est strictement interdit de fumer dans l'avion, y compris dans les toilettes.</td>
      <td>[SPEAKER_00][fr] Pour detacher votre ceinture, soulevez la partie superieure de la boucle.<br>Il est strictement interdit de fumer dans l'avion, y compris dans les toilettes.</td>
      <td>[SPEAKER_UNKNOWN][45.55-55.84] Pour detacher votre ceinture, soulevez la partie superieure de la boucle. Il est strictement interdit de fumer dans l'avion, y compris dans les toilettes.</td>
    </tr>
    <tr>
      <td><code>EN smoking</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] This is a <b>no-smoking</b> flight, and it is strictly prohibited to smoke in the toilets.</td>
      <td>[SPEAKER_01] This is a <b>no smoking</b> flight, and it is strictly prohibited to smoke in the toilets.</td>
      <td>[SPEAKER_01][en] This is a <b>no-smoking</b> flight, and it is strictly prohibited to smoke in the toilets.</td>
      <td>[SPEAKER_01][56.12-61.19] This is a <b>no smoking</b> flight, and it is strictly prohibited to smoke in the toilets.</td>
    </tr>
    <tr>
      <td><code>FR oxygen</code></td>
      <td>En cas de depressurisation, un masque a oxygene tombera automatiquement a votre portee.<br><b>Tirez</b> sur le masque pour liberer l'oxygene, placez-le sur votre visage.</td>
      <td>[SPEAKER_00][fr] En cas de depressurisation, un masque a oxygene tombera automatiquement a votre portee.<br><b>Tirer</b> sur le masque pour liberer l'oxygene, placez-le sur votre visage.</td>
      <td>[SPEAKER_00] En cas de <b>depressionisation</b>, un masque a oxygene tombera automatiquement a votre portee.</td>
      <td>[SPEAKER_00][fr] En cas de depressurisation, un masque a oxygene tombera automatiquement a votre portee.<br><b>Tirez</b> sur le masque pour liberer l'oxygene. <b>Placez</b>-le sur votre visage.</td>
      <td>[SPEAKER_UNKNOWN][61.38-67.52] En cas de depressurisation, un masque a oxygene tombera automatiquement a votre portee.<br>[SPEAKER_UNKNOWN][73.47-78.38] Tirez sur le masque pour liberer l'oxygene, placez-le sur votre visage.</td>
    </tr>
    <tr>
      <td><code>EN oxygen</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] If there is a sudden decrease in cabin pressure, your oxygen mask will drop automatically in front of you.<br>Pull the mask toward you to start the flow of oxygen. Place the mask over your nose and mouth. Make sure your own mask is well adjusted before helping others.</td>
      <td>[SPEAKER_01] If there is a sudden decrease in cabin pressure, your oxygen mask will drop automatically in front of you</td>
      <td>[SPEAKER_01][en] If there is a sudden decrease in cabin pressure, your oxygen mask will drop automatically in front of you.<br>Pull the mask toward you to start the flow of oxygen. Place the mask over your nose and mouth. Make sure your own mask is well adjusted before helping others.</td>
      <td>[SPEAKER_01][67.62-73.41] If there is a sudden decrease in cabin pressure, your oxygen mask will drop automatically in front of you.<br>[SPEAKER_01][78.65-87.45] Pull the mask toward you to start the flow of oxygen. Place the mask over your nose and mouth. Make sure your own mask is well adjusted before helping others.</td>
    </tr>
    <tr>
      <td><code>FR evacuation / exits</code></td>
      <td>Une fois votre masque ajuste, il vous sera possible d'aider d'autres personnes.<br>En cas d'evacuation, des panneaux lumineux EXIT vous permettent de localiser les issues de secours.<br>Reperez maintenant le panneau EXIT le plus proche de votre siege.<br>Il peut se trouver derriere vous.<br>Les issues de secours sont situees de chaque cote de la cabine, a l'avant, au centre, a l'arriere.</td>
      <td>[SPEAKER_00][fr] Une fois votre masque ajuste, il vous sera possible d'aider d'autres personnes.<br>En cas d'evacuation, des panneaux lumineux EXIT vous permettent de localiser les issues de secours.<br>Reperez maintenant le panneau EXIT le plus proche de votre siege.<br>Il peut se trouver derriere vous.<br>Les issues de secours sont situees de chaque cote de la cabine, a l'avant, au centre, a l'arriere.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_00][fr] Une fois votre masque ajuste, il vous sera possible d'aider d'autres personnes.<br>En cas d'evacuation, des panneaux lumineux <b>"exit"</b> vous permettent de localiser les issues de secours.<br>Reperez maintenant le panneau <b>"exit"</b> le plus proche de votre siege.<br>Il peut se trouver derriere vous.<br>Les issues de secours sont situees de chaque cote de la cabine, a l'avant, au centre, a l'arriere.</td>
      <td>[SPEAKER_UNKNOWN][87.45-98.54] Une fois votre masque ajuste, il vous sera possible d'aider d'autres personnes. En cas d'evacuation, des panneaux lumineux <b>exit</b> vous permettent de localiser les issues de secours.<br>[SPEAKER_UNKNOWN][98.54-104.51] Reperez maintenant le panneau <b>exit</b> le plus proche de votre siege. Il peut se trouver derriere vous.<br>[SPEAKER_UNKNOWN][116.05-123.53] Les issues de secours sont situees de chaque cote de la cabine, a l'avant, au centre, a l'arriere.</td>
    </tr>
    <tr>
      <td><code>EN exits</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] In case of an emergency, the illuminated exit signs will help you locate the exit doors.<br>Please take a moment now to locate the exit nearest you.<br>The nearest exit may be behind you.<br>Emergency exits on each side of the cabin are located at the front, in the center, and at the rear.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] In case of an emergency, the illuminated exit signs will help you locate the exit doors.<br>Please take a moment now to locate the exit nearest you.<br>The nearest exit may be behind you.<br>Emergency exits on each side of the cabin are located at the front, in the center, and at the rear.</td>
      <td>[SPEAKER_01][104.51-115.87] In case of an emergency, the illuminated exit signs will help you locate the exit doors. Please take a moment now to locate the exit nearest you. The nearest exit may be behind you.<br>[SPEAKER_01][124.12-130.53] Emergency exits on each side of the cabin are located at the front, in the center, and at the rear.</td>
    </tr>
    <tr>
      <td><code>FR evacuation / slides</code></td>
      <td>Pour evacuer l'avion, suivez le marquage lumineux.<br>Les portes seront ouvertes par l'equipage.<br>Les toboggans se deploient automatiquement.</td>
      <td>[SPEAKER_00][fr] Pour evacuer l'avion, suivez le marquage lumineux.<br>Les portes seront ouvertes par l'equipage.<br>Les toboggans se deploient automatiquement.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_00][fr] Pour evacuer l'avion, suivez le marquage lumineux.<br>Les portes seront ouvertes par l'equipage.<br>Les toboggans se deploient automatiquement.</td>
      <td>[SPEAKER_UNKNOWN][130.95-134.52] Pour evacuer l'avion, suivez le marquage lumineux.<br>[SPEAKER_UNKNOWN][140.12-143.17] Les portes seront ouvertes par l'equipage.<br>[SPEAKER_UNKNOWN][145.59-148.27] Les toboggans se deploient automatiquement.</td>
    </tr>
    <tr>
      <td><code>EN evacuation / slides</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] In the event of an evacuation, pathway lighting on the floor will guide you to the exits.<br>Doors will be opened by the cabin crew.<br>The emergency slides will automatically inflate.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] In the event of an evacuation, pathway lighting on the floor will guide you to the exits.<br>Doors will be opened by the cabin crew.<br>The emergency slides will automatically inflate.</td>
      <td>[SPEAKER_01][134.57-139.91] In the event of an evacuation, pathway lighting on the floor will guide you to the exits.<br>[SPEAKER_01][143.25-145.59] Doors will be opened by the cabin crew.<br>[SPEAKER_01][148.31-151.14] The emergency slides will automatically inflate.</td>
    </tr>
    <tr>
      <td><code>FR life jacket</code></td>
      <td>Le gilet de sauvetage est situe sous votre siege ou dans <b>la coudoir</b> centrale.<br><b>Passez</b> la tete dans l'encolure, attachez et serrez les sangles.<br>Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.</td>
      <td>[SPEAKER_00][fr] Le gilet de sauvetage est situe sous votre siege ou dans <b>la coudoir</b> centrale.<br><b>Passez</b> la tete dans l'encolure, attachez et serrez les sangles.<br>Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_00][fr] Le gilet de sauvetage est situe sous votre siege ou dans <b>l'accoudoir</b> central.<br><b>Passez</b> la tete dans l'encolure, attachez et serrez les sangles.<br>Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.</td>
      <td>[SPEAKER_UNKNOWN][151.14-156.31] Le gilet de sauvetage est situe sous votre siege ou dans <b>la coude-oreille</b> centrale.<br>[SPEAKER_UNKNOWN][159.95-164.62] <b>Placez</b> la tete dans <b>l'encadrement</b>, attachez et serrez les sangles.<br>[SPEAKER_UNKNOWN][172.25-178.15] Une fois a l'exterieur de l'avion, gonflez votre gilet en tirant sur les poignees rouges.</td>
    </tr>
    <tr>
      <td><code>EN life jacket</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] Your life jacket is under your seat or in the central armrest.<br>Place it over your head and pull the straps tightly around your waist. Inflate your life jacket by pulling the red toggles.<br>Do this only when you are outside the aircraft.</td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] Your life jacket is under your seat or in the central armrest.<br>Place it over your head and pull the straps tightly around your waist. Inflate your life jacket by pulling the red toggles.<br>Do this only when you are outside the aircraft.</td>
      <td>[SPEAKER_01][156.53-159.95] Your life jacket is under your seat or in the central armrest.<br>[SPEAKER_01][164.83-172.18] Place it over your head and pull the straps tightly around your waist. Inflate your life jacket by pulling the red toggles.<br>[SPEAKER_01][178.35-181.50] Do this only when you are outside the aircraft.</td>
    </tr>
    <tr>
      <td><code>FR takeoff</code></td>
      <td>Nous allons bientot decoller.<br>La tablette doit etre rangee et votre dossier redresse.</td>
      <td>[SPEAKER_00][fr] Nous allons bientot decoller.<br>La tablette doit etre rangee et votre dossier redresse.</td>
      <td>[SPEAKER_00] Nous allons bientot decoller.<br>La tablette doit etre rangee et votre dossier redresse.</td>
      <td>[SPEAKER_00][fr] Nous allons bientot decoller.<br>La tablette doit etre rangee et votre dossier redresse.</td>
      <td>[SPEAKER_UNKNOWN][184.64-190.98] Nous allons bientot decoller. La tablette doit etre rangee et votre dossier redresse.</td>
    </tr>
    <tr>
      <td><code>EN takeoff</code></td>
      <td><b>[MISSING]</b></td>
      <td>[SPEAKER_01][en] In preparation for <b>take-off</b>, please make sure your tray table is stowed and secure and that your <b>seat back</b> is in the upright position.</td>
      <td>[SPEAKER_01] In preparation for <b>takeoff</b>, please make sure your tray table is stowed and secure, and that your <b>seatback</b> is in the upright position.</td>
      <td>[SPEAKER_01][en] In preparation for <b>takeoff</b>, please make sure your tray table is stowed and secure, and that your <b>seatback</b> is in the upright position.</td>
      <td>[SPEAKER_01][191.15-198.64] In preparation for <b>takeoff</b>, please make sure your tray table is stowed and secure, and that your <b>seatback</b> is in the upright position.</td>
    </tr>
    <tr>
      <td><code>FR closing turns</code></td>
      <td>L'usage des appareils electroniques est interdit pendant le decollage et l'atterrissage.<br>Les telephones portables doivent rester eteints pendant tout le vol.<br>Une notice de securite placee devant vous est a votre disposition.<br>[SPEAKER_00] Merci pour votre attention. Nous vous souhaitons un bon vol.</td>
      <td>[SPEAKER_00][fr] L'usage des appareils electroniques est interdit pendant le decollage et l'atterrissage.<br>[SPEAKER_00][fr] Les telephones portables doivent rester eteints pendant tout le vol.<br>[SPEAKER_00][fr] Une notice de securite placee devant vous est a votre disposition.<br>[SPEAKER_00][fr] Merci pour votre attention. Nous vous souhaitons un bon vol.</td>
      <td>[SPEAKER_00] L'usage des appareils electroniques est interdit pendant le decollage et l'atterrissage.<br>[SPEAKER_00] Les telephones portables doivent rester eteints pendant tout le vol.<br>[SPEAKER_00] Une notice de securite placee devant vous est a votre disposition.<br>[SPEAKER_00] Merci pour votre attention. Nous vous souhaitons un bon vol.</td>
      <td>[SPEAKER_00][fr] L'usage des appareils electroniques est interdit pendant le decollage et l'atterrissage.<br>[SPEAKER_00][fr] Les telephones portables doivent rester eteints pendant tout le vol.<br>[SPEAKER_00][fr] Une notice de securite placee devant vous est a votre disposition.<br>[SPEAKER_00][fr] Merci pour votre attention. Nous vous souhaitons un bon vol.</td>
      <td>[SPEAKER_UNKNOWN][198.64-203.93] L'usage des appareils electroniques est interdit pendant le decollage et l'atterrissage.<br>[SPEAKER_UNKNOWN][208.55-212.29] Les telephones portables doivent rester eteints pendant tout le vol.<br>[SPEAKER_UNKNOWN][216.55-221.05] Une notice de securite placee devant vous est a votre disposition.<br>[SPEAKER_UNKNOWN][226.51-230.37] Merci pour votre attention. Nous vous souhaitons un bon vol.</td>
    </tr>
    <tr>
      <td><code>EN closing turns</code></td>
      <td>[SPEAKER_01] We encourage everyone to read the safety information leaflet located in the <b>seat back</b> pocket.<br>[SPEAKER_01] Thank you for your attention. We wish you a very pleasant flight.</td>
      <td>[SPEAKER_01][en] The use of electronic devices is prohibited during <b>take-off</b> and landing.<br>[SPEAKER_01][en] Mobile phones must remain switched off for the duration of the flight.<br>[SPEAKER_01][en] We encourage everyone to read the safety information leaflet located in the <b>seat back</b> pocket.<br>[SPEAKER_01][en] Thank you for your attention. We wish you a very pleasant flight.</td>
      <td>[SPEAKER_01] The use of electronic devices is prohibited during <b>takeoff</b> and landing.<br>[SPEAKER_01] Mobile phones must remain switched off for the duration of the flight.<br>[SPEAKER_01] We encourage everyone to read the safety information leaflet located in the <b>seatback</b> pocket.<br>[SPEAKER_01] Thank you for your attention. We wish you a very pleasant flight.</td>
      <td>[SPEAKER_01][en] The use of electronic devices is prohibited during <b>takeoff</b> and landing.<br>[SPEAKER_01][en] Mobile phones must remain switched off for the duration of the flight.<br>[SPEAKER_01][en] We encourage everyone to read the safety information leaflet located in the <b>seat back</b> pocket.<br>[SPEAKER_01][en] Thank you for your attention. We wish you a very pleasant flight.</td>
      <td>[SPEAKER_01][204.08-208.37] The use of electronic devices is prohibited during <b>takeoff</b> and landing.<br>[SPEAKER_01][212.30-216.37] Mobile phones must remain switched off for the duration of the flight.<br>[SPEAKER_01][221.21-226.38] We encourage everyone to read the safety information leaflet located in the <b>seatback</b> pocket.<br>[SPEAKER_01][230.51-234.91] Thank you for your attention. We wish you a very pleasant flight.</td>
    </tr>
  </tbody>
</table>

## Observations

- Whisper-3-Large in naive mode mostly stays in French through the central sections and drops most English alternation.
- Whisper-3-Large-Verbatim recovers the bilingual alternation cleanly, although the French armrest phrase is still slightly degraded as `la coudoir centrale`.
- Qwen3-ASR in naive mode is much less stable on this sample and drops the entire life-jacket section in the current run.
- Qwen3-ASR-Verbatim recovers the full bilingual sequence for the life-jacket excerpt after the sentence-splitting fixes, and preserves the correct `l'accoudoir central` wording.
- VibeVoice-ASR is competitive on coverage and preserves the bilingual alternation, but its French wording is noisier in this excerpt, including `coude-oreille centrale` and `encadrement`.
- Qwen still has a backend-level limitation on some mixed-language timing ranges: when a pass is misanchored, the forced aligner can snap later speech to the start of the window.
