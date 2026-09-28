# GFlava-Quant -- lokales Web-Tool zur Modell-Quantisierung

*[Read this in English](README.md)*

Eine kleine Oberflaeche im Browser fuer das, was wir bisher per PowerShell
gemacht haben: bf16-Modell auswaehlen, Format waehlen, quantisieren lassen.
Oben rechts: Sprachmenue (Deutsch/English, komplette Oberflaeche wird
uebersetzt), ein Dark-/Light-Mode-Umschalter -- beides wird im Browser
gespeichert (localStorage) und bleibt bei einem Neuladen erhalten -- und ein
Zahnrad fuer die Einstellungen (Modell-Ordner, ComfyUI-Pfad, Systemcheck,
siehe unten).

**Wichtig:** Das ist eine *lokale* Anwendung, kein Hosting-Dienst. Sie laeuft
nur auf deinem eigenen PC, liest/schreibt nur Dateien auf deinem PC und ruft
`ctq` auf deinem PC auf. Der Server ist nur unter `127.0.0.1` (also nur von
diesem Rechner aus) erreichbar.

## Voraussetzung (einmalig)

Du brauchst `convert_to_quant` (ctq, das eigentliche Quantisierungs-Programm)
und `flask` (fuer die Web-Oberflaeche). `start_server.bat` (siehe unten)
prueft beide beim Start automatisch und installiert sie auf Wunsch; von
Hand geht es auch:

```powershell
& "H:\ComfyUI_windows_portable\ComfyUI-Easy-Install\python_embeded\python.exe" -m pip install flask convert_to_quant
```

## Starten

**Am einfachsten: `start_server.bat` doppelklicken** (Englisch, damit sie
auch auf einem fremden Rechner verstaendlich ist). Sie:

1. sucht `python.exe` und `quant_server.py` zuerst automatisch (falls die
   `.bat` im selben Ordner wie `python.exe` liegt bzw. `quant_server.py`
   daneben liegt) -- findet sie das nicht, oeffnet sich ein Datei-Auswahl-
   Fenster, in dem du beides manuell auswaehlen kannst;
2. merkt sich beide Pfade danach in `start_server_config.txt` neben der
   `.bat`, damit du sie auf diesem Rechner nur einmal auswaehlen musst
   (Datei loeschen, um die Auswahl zu wiederholen -- z. B. auf einem
   anderen Server, wo `python.exe` woanders liegt);
3. prueft, ob `flask` und `convert_to_quant` (ctq) installiert sind -- falls
   nicht, erklaert sie jeweils kurz wofuer es gebraucht wird und fragt, ob
   es installiert werden soll (Ja/Nein);
4. startet den Server und oeffnet nach 2 Sekunden automatisch den Browser
   unter **http://127.0.0.1:8877**.

Das Fenster mit dem laufenden Server einfach offen lassen, solange du die
Seite benutzt. Mit `Strg+C` oder Fenster schliessen beendest du den Server
wieder.

Alternativ von Hand ueber PowerShell, z. B. wenn du `quant_server.py` nach
`H:\ComfyUI_windows_portable\ComfyUI-Easy-Install\python_embeded\` gelegt
hast:

```powershell
& "H:\ComfyUI_windows_portable\ComfyUI-Easy-Install\python_embeded\python.exe" "H:\ComfyUI_windows_portable\ComfyUI-Easy-Install\python_embeded\quant_server.py"
```

## Benutzung

1. **Modell auswaehlen**: Dropdown mit allen `.safetensors`-Dateien aus den
   Modell-Ordnern, die unter dem Zahnrad-Symbol oben rechts ("Einstellungen")
   eingetragen sind. Beim allerersten Start sind das drei Standard-Ordner
   (`E:\_LLM\_ComfyOutput\diffusion_models`,
   `E:\_LLM\_Modele\comfyui_models\models\diffusion_models`,
   `E:\_LLM\_Modele\comfyui_models\models\StableDiffusion`), danach kannst du
   dort beliebige eigene Ordner hinzufuegen, umbenennen oder entfernen --
   ueber "Durchsuchen" oeffnet sich ein echter Windows-Ordnerdialog (der
   Server steuert dafuer kurz ein natives Auswahlfenster an, da ein
   normales `<input type=file>` aus Sicherheitsgruenden keinen echten
   Dateisystempfad verraet) oder per Texteingabe. Jede Aenderung wird
   dauerhaft in `config.json` neben `quant_server.py` gespeichert und ist
   beim naechsten Start automatisch wieder vorausgewaehlt. Jede
   Modell-Option zeigt Unterordner, Dateiname und Groesse, gruppiert nach
   Herkunfts-Ordner; nach der Auswahl steht der volle Pfad nochmal als
   Vorschau darunter, und falls der Modelltyp erkannt wird (siehe Punkt 5),
   erscheint direkt darunter eine farbige Pille dazu. Der &#8635;-Button
   liest die Ordner erneut ein (z. B. nach einem neuen Merge/Download).
   Alternativ ueber den Link "Pfad stattdessen manuell eingeben" einen
   beliebigen Pfad eintippen (fuer manuell eingegebene Pfade laeuft keine
   automatische Erkennung).
2. **Ausgabedatei**: kannst du leer lassen, dann wird automatisch
   `<dateiname>_<format>.safetensors` im selben Ordner vorgeschlagen.
3. **Quantisierungsformat**: `INT8 ConvRot` ist vorausgewaehlt -- das haben
   wir gemeinsam Ende-zu-Ende getestet und in ComfyUI ohne Warnung geladen.
   Bei ConvRot ist die Gruppengroesse (Standard 256) laut `ctq`-Dokumentation
   nicht modellspezifisch und muss eine **Potenz von 4** sein (4, 16, 64,
   256, 1024 -- 128 ist trotz Zweierpotenz ungueltig). Bei `INT8 Block-Wise`
   gibt es analog ein Feld fuer die Block-Groesse (Standard 128). Die
   anderen Formate ruft das Tool laut ctq-eigener Dokumentation korrekt auf,
   aber wir haben ihr Marker-Format nicht gegen eine offizielle Datei
   verglichen -- probier sie, aber prüf das Ergebnis in ComfyUI, bevor du dich
   darauf verlaesst.
4. **Layer-Ausschluss**: Ueber die Schnellauswahl-Kacheln oder direkt im
   Dropdown darunter -- beide zeigen jetzt **alle** Modelltypen, die ctq
   kennt: alle 24 Presets aus `ctq --help-filters` deiner installierten
   Version (Bild-, Video-, Diffusions- und Text-Encoder-Modelle) plus unsere
   3 eigenen Community-Regexes (Qwen-Image 2.1 Single-Stream, Flux.1,
   SDXL/Illustrious). `Qwen-Image-2.1 (Single-Stream)` und `ctq --anima`
   sind von uns Ende-zu-Ende in ComfyUI getestet (gruener
   "verifiziert"/"Ende-zu-Ende getestet"-Badge). Flux.1 und SDXL/Illustrious
   sind aus oeffentlich dokumentierten Architektur-Konventionen abgeleitet,
   aber nicht von uns selbst verifiziert (gelber Badge) -- Ergebnis einmal in
   ComfyUI pruefen. Bei allen ctq-eigenen Presets zeigt der Hinweistext
   direkt die Original-Beschreibung aus `ctq --help-filters` deiner
   installierten Version. Unter "Erweitert" steht zum ausgewaehlten Preset
   passend, ob zusaetzliche ctq-Argumente noetig sind.
5. **Automatische Modelltyp-Erkennung**: Sobald du oben ein Modell aus der
   Liste auswaehlst, liest der Server im Hintergrund nur den Tensor-Header
   der Datei (keine Tensor-Daten, daher auch bei sehr grossen Modellen
   schnell) und prueft die Tensor-Namen gegen Architektur-Signaturen. Passt
   eine eindeutig, erscheint eine farbige Pille ("Erkannt: Anima" o. ae.)
   und der passende Layer-Ausschluss aus Punkt 4 wird automatisch gesetzt --
   du kannst das jederzeit ueberschreiben. Erkannt werden Anima, Flux.1,
   Flux.2, SDXL/Illustrious, Qwen-Image 2.1 (Single-Stream), ctq's aeltere
   Qwen-Dual-Stream-Variante, Z-Image (+ Refiner), Wan, HunyuanVideo, Krea2,
   Boogu, Ideogram4, Radiance, NeRF (gross/klein), Chroma/distilled
   (gross/klein), MinimaxH3, LTXv2, Gemma4 und Qwen3-VL. Die Signaturen fuer
   die ctq-eigenen Presets stammen direkt aus den `MODEL_FILTERS`-Konstanten
   deiner installierten `convert_to_quant`-Version (keine Vermutung
   unsererseits); Anima, Flux.1, Flux.2, SDXL/Illustrious und Qwen-Image 2.1
   wurden zusaetzlich gegen echte Dateien auf deinem Rechner verifiziert.
   Flux.1 wird ueber `double_blocks`/`single_blocks` mit `img_mod`/`txt_mod`
   erkannt (Flux.2 hat stattdessen `stream_modulation` und kein `img_mod`),
   SDXL/Illustrious ueber den klassischen U-Net-Aufbau
   (`input_blocks`/`middle_block`/`output_blocks`/`label_emb` -- kommt bei
   keiner DiT-Architektur wie Flux/Qwen/Anima vor, daher besonders sicher zu
   unterscheiden). LENS bleibt bewusst **ohne** automatische Erkennung -- es
   teilt sich zu viele Tensor-Namen mit ctq's Qwen-Dual-Stream-Preset, ohne
   ein eigenes eindeutiges Merkmal; lieber keine Erkennung als eine falsche.
   Passt nichts eindeutig, bleibt die Pille einfach weg und du waehlst wie
   bisher manuell. Fuer manuell eingegebene Pfade (statt Auswahl aus der
   Liste) laeuft keine automatische Erkennung.
6. **Erweitert -&gt; Low-Memory-Modus**: steuert ctq's `--low-memory`-Flag.
   Wir haben im installierten ctq-Quellcode nachgeprueft: Das betrifft
   ausschliesslich den **System-RAM** beim Einlesen (alle Gewichte vorab in
   den RAM laden vs. einzeln von der Platte nachladen) -- **nicht den VRAM**
   der Grafikkarte, die GPU-Verarbeitung verschiebt bei ctq ohnehin immer nur
   einen Tensor auf einmal und gibt ihn danach wieder frei. "Automatisch"
   (Standard) aktiviert das Flag nur, wenn die Eingabedatei mehr als 50&nbsp;%
   des gerade verfuegbaren System-RAM belegt -- das ist ctq's eigene
   Empfehlung laut `ctq --help`. Kleinere Modelle werden dadurch ohne
   Streaming-Overhead schneller verarbeitet. Die tatsaechlich gemessenen
   Werte und die Entscheidung stehen im Log jedes Laufs; ueber "Immer an"/
   "Immer aus" laesst sich das Flag auch manuell erzwingen.
7. **Quantisieren starten** klicken. Ein Fly-out-Fenster oeffnet sich mit
   Status, Dauer, Fortschrittsbalken und Live-Log, bis der Job fertig ist
   oder ein Fehler auftritt. Ueber das &times; kannst du es minimieren --
   der Job laeuft im Hintergrund weiter, eine kleine Pille unten rechts
   zeigt den aktuellen Status und oeffnet das Fenster per Klick wieder.

Bei `INT8 ConvRot` und `INT8 Tensor-Wise` laeuft danach automatisch der
Marker-Fix (kuerzt die `.comfy_quant`-Marker auf genau die Felder, die
ComfyUI erwartet), damit das Ergebnis ohne die `unet unexpected`-Warnung
laedt -- also genau der Workflow, den wir uns gemeinsam erarbeitet haben,
nur nicht mehr Schritt fuer Schritt von Hand.

## Logs

Jeder Quantisierungs-Lauf (voller ctq-Befehl + komplette Ausgabe) wird
dauerhaft als Datei unter `logs/` neben `quant_server.py` gespeichert --
bleibt also auch nach einem Server-Neustart erhalten. Im Abschnitt
"Fruehere Laeufe" auf der Seite kannst du jeden gespeicherten Lauf jederzeit
nachtraeglich nachlesen.

## Einstellungen &amp; Systemcheck

Zahnrad oben rechts oeffnet die Einstellungen:

- **Modell-Ordner**: siehe oben -- beliebig viele, frei waehlbar, dauerhaft
  gespeichert.
- **ComfyUI-Installation**: Pfad zu deinem ComfyUI-Ordner (nicht
  `python_embeded`, sondern der Ordner mit `custom_nodes` darin). Wird beim
  allerersten Start anhand des Python-Interpreters geraten (funktioniert bei
  der ueblichen ComfyUI-Portable-Ordnerstruktur automatisch), laesst sich
  aber jederzeit aendern. Wird nur fuer den Systemcheck unten gebraucht.
- **ctq-Programm**: wird normalerweise automatisch neben deinem Python
  gefunden (siehe Kopfzeile). Nur falls das fehlschlaegt, kannst du hier
  manuell den Pfad zu `ctq.exe` eintragen.
- **Systemcheck**: prueft (braucht Internetzugriff)
  - ob `convert_to_quant` (ctq) ueberhaupt installiert ist. Fehlt es
    komplett (z. B. auf einem neuen Rechner, wo nur `flask` ueber
    `start_server.bat` installiert wurde, aber `ctq` noch nie), installiert
    ein "Installieren"-Knopf es per `pip install convert_to_quant`. Ist es
    installiert, aber veraltet (Vergleich gegen PyPI via
    `pip index versions`), holt "Aktualisieren" die neueste Version --
    beides derselbe `pip install -U convert_to_quant`-Befehl.
  - ob das Custom-Node [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)
    in `<ComfyUI>\custom_nodes\` installiert ist, und falls ja, ob es ein
    Update gibt (Vergleich der lokalen gegen die entfernte Git-Revision).
    Fehlt es, klont ein "Installieren"-Knopf das Repository per `git clone`
    in deinen `custom_nodes`-Ordner; ist es installiert und veraltet, holt
    "Aktualisieren" per `git pull` die neueste Version. Beides ist ein
    echter, sofort ausgefuehrter Git-Befehl auf deinem Rechner -- also nur
    klicken, wenn du das wirklich willst. Git muss dafuer installiert und im
    `PATH` sein.

Alle Einstellungen liegen in `config.json` neben `quant_server.py` -- die
Datei kannst du auch von Hand bearbeiten oder sichern.

## Grenzen

- Es kann immer nur ein Quantisierungs-Job gleichzeitig laufen (ein zweiter
  Klick auf "Quantisieren starten" waehrend ein Job laeuft, gibt einen
  Fehler statt zwei Jobs parallel zu starten -- das wuerde sich sonst um
  RAM/VRAM streiten).
- Eingestellte Formularwerte (Format, Exclude-Preset, Extra-Argumente)
  werden zwischen zwei Server-Neustarts nicht gespeichert -- nur die
  Logs unter `logs/` sowie die Einstellungen in `config.json` bleiben
  dauerhaft erhalten.
