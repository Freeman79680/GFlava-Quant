# GFlava-Quant – lokales Web-Tool zur Modell-Quantisierung

*[Read this in English](README.md)*

Eine kleine Oberfläche im Browser für das, was sonst per PowerShell und
`ctq`-Befehlen passiert: bf16-Modell auswählen, Format wählen, quantisieren
lassen. Oben rechts: Sprachmenü (Deutsch/English, die komplette Oberfläche
wird übersetzt), ein Dark-/Light-Mode-Umschalter, ein Schalter für das
**Minimal-Design** (keine Verläufe, nur eine Akzentfarbe) und ein Zahnrad für
die Einstellungen (Modell-Ordner, ComfyUI-Pfad, Updates und Systemcheck, siehe
unten; klappt als Dropdown auf). Sprache, Theme und Minimal-Design werden im
Browser gespeichert (localStorage) und bleiben bei einem Neuladen erhalten.

**Wichtig:** Das ist eine *lokale* Anwendung, kein Hosting-Dienst. Sie läuft
nur auf deinem eigenen PC, liest und schreibt nur Dateien auf deinem PC und
ruft `ctq` auf deinem PC auf. Der Server ist nur unter `127.0.0.1` (also nur
von diesem Rechner aus) erreichbar.

## Aufbau

Seit Version 1.3 besteht GFlava-Quant aus mehreren Dateien, die zusammen in
einem Ordner liegen müssen:

```
quant_server.py        Server (Flask), ctq-Aufruf, Warteschlange, Erkennung
restore_bf16.py        "Zurück nach bf16": entpackt INT8-Modelle wieder
templates/index.html   Gerüst der Seite
static/style.css       Design (Dunkel, Hell, Minimal)
static/i18n.js         alle Texte auf Deutsch und Englisch
static/app.js          Logik der Oberfläche
fonts/, icon.png       Schriften und Logo
start_server.bat       Starter für Windows
start_server.sh        Starter für Linux und macOS
tests/                 automatische Tests
```

Fehlen `templates` oder `static`, meldet das `start_server.bat` beim Start,
und die Seite zeigt statt der Oberfläche einen Hinweis.

## Voraussetzung (einmalig)

Du brauchst `convert_to_quant` (ctq, das eigentliche
Quantisierungs-Programm) und `flask` (für die Web-Oberfläche).
`start_server.bat` (siehe unten) prüft beide beim Start automatisch und
installiert sie auf Wunsch. Von Hand geht es auch, mit dem Python, das dein
ComfyUI benutzt:

```powershell
& "C:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install flask convert_to_quant
```

## Starten

**Am einfachsten: `start_server.bat` doppelklicken** (Englisch, damit sie
auch auf einem fremden Rechner verständlich ist). Sie:

1. sucht `python.exe` und `quant_server.py` zuerst automatisch (falls die
   `.bat` im selben Ordner wie `python.exe` liegt bzw. `quant_server.py`
   daneben liegt). Findet sie das nicht, öffnet sich ein Datei-Auswahlfenster,
   in dem du beides von Hand auswählst;
2. merkt sich beide Pfade danach in `start_server_config.txt` neben der
   `.bat`, damit du sie auf diesem Rechner nur einmal auswählen musst (Datei
   löschen, um die Auswahl zu wiederholen);
3. prüft, ob `templates` und `static` neben `quant_server.py` liegen;
4. prüft, ob `flask` und `convert_to_quant` (ctq) installiert sind. Falls
   nicht, erklärt sie jeweils kurz, wofür es gebraucht wird, und fragt, ob es
   installiert werden soll (Ja/Nein);
5. startet den Server und öffnet automatisch den Browser, sobald der Server bereit ist,
   unter **http://127.0.0.1:8877**.

Das Fenster mit dem laufenden Server einfach offen lassen, solange du die
Seite benutzt. Mit `Strg+C` oder Fenster schließen beendest du den Server.

Alternativ von Hand über PowerShell:

```powershell
& "C:\ComfyUI_windows_portable\python_embeded\python.exe" "C:\Pfad\zu\GFlava-Quant\quant_server.py"
```

**Linux / macOS:** `./start_server.sh` starten. Es nimmt `venv/bin/python` daneben oder
`python3`; für deine ComfyUI-Umgebung z. B.
`GFLAVA_PYTHON=~/ComfyUI/venv/bin/python ./start_server.sh`. Das Skript bietet an,
`flask` und `convert_to_quant` zu installieren, startet den Server und öffnet den Browser.

## Benutzung

Die Seite ist eine einzige Ansicht mit nummerierten Schritten. Einsteiger
kommen mit Schritt 1 und dem Knopf „Empfohlene Einstellungen verwenden“ aus,
Details stehen hinter „Erweitert“ und „Mehr dazu“.

1. **Modell auswählen**: durchsuchbare Liste mit allen `.safetensors`-Dateien
   aus den Modell-Ordnern, die unter dem Zahnrad eingetragen sind. Beim
   allerersten Start ist die Liste leer und die Einstellungen öffnen sich
   automatisch mit einem Hinweis. Beim Hinzufügen eines Ordners öffnet sich
   ein echter Windows-Ordnerdialog (ein normales `<input type=file>` verrät
   aus Sicherheitsgründen keinen echten Dateipfad). Der &#8635;-Knopf liest die
   Ordner neu ein, zum Beispiel nach einem Download. Die Liste lässt sich nach Name
   oder Änderungsdatum sortieren, aufsteigend oder absteigend (die Wahl merkt sich
   der Browser). Über „Pfad stattdessen von
   Hand eingeben“ geht auch ein beliebiger Pfad, über „Ausgabedatei ändern“ ein
   eigener Zielname. Ohne Angabe wird `<dateiname>_<format>.safetensors` im
   selben Ordner verwendet.
2. **So viel kleiner wird die Datei**: Das Bit-Raster neben der Auswahl zeigt
   bf16 gegen das gewählte Format. Sobald ein Modell gewählt ist, rechnet die
   Schätzung mit den echten Tensor-Größen aus dem Datei-Header. Es bleibt eine
   Schätzung aus der Bitbreite: geschützte Layer bleiben größer, die echte
   Datei liegt deshalb etwas darüber. **Empfohlene Einstellungen verwenden**
   setzt INT8 ConvRot und den Layer-Schutz für die erkannte Architektur.
3. **Format** (Schritt 2): `INT8 ConvRot` ist vorausgewählt. Das ist Ende zu
   Ende getestet und lädt in ComfyUI ohne Warnung. Bei ConvRot muss die
   Gruppengröße (Standard 256) laut `ctq`-Dokumentation eine **Potenz von 4**
   sein (4, 16, 64, 256, 1024). Bei `INT8 Block-Wise` gibt es ein Feld für
   die Block-Größe (Standard 128). Die anderen Formate (INT8 Tensor-Wise,
   FP8, NVFP4, MXFP8) ruft das Tool laut ctq-eigener Dokumentation auf, sie
   sind aber nicht von uns gegen offizielle Dateien geprüft. Ergebnis in
   ComfyUI kontrollieren, bevor du dich darauf verlässt.
4. **Layer-Schutz** (Schritt 3): legt fest, welche empfindlichen Layer
   unquantisiert bleiben. Zur Auswahl stehen alle Presets aus
   `ctq --help-filters` deiner installierten Version plus vier eigene
   Regexes (Qwen-Image 2.1 Single-Stream, Flux.1, SDXL/Illustrious,
   MOSS-Image), insgesamt 28. Der MOSS-Image-Regex ist aus den veröffentlichten
   Tensor-Namen abgeleitet und noch nicht mit echten Gewichten getestet. Dafür
   braucht es eine einzelne `.safetensors`-Datei (das `comfyui/export.py` des
   Projekts fasst die heruntergeladenen Teile zusammen). Bei den ctq-eigenen Presets zeigt der Hinweistext die
   Original-Beschreibung aus `ctq --help-filters`.
5. **Automatische Modelltyp-Erkennung**: Beim Auswählen eines Modells liest
   der Server nur den Tensor-Header der Datei (keine Tensor-Daten, daher auch
   bei sehr großen Modellen schnell) und vergleicht die Tensor-Namen mit
   bekannten Architektur-Signaturen. Passt eine eindeutig, erscheint eine
   farbige Pille („Erkannt: Anima“ o. ä.) und der passende Layer-Schutz wird
   gesetzt. Du kannst das jederzeit überschreiben. Erkannt werden Anima,
   Flux.1, Flux.2, SDXL/Illustrious, Qwen-Image 2.1 (Single-Stream),
   MOSS-Image (Base und SFT), ctqs
   ältere Qwen-Dual-Stream-Variante, Z-Image (+ Refiner), Wan, HunyuanVideo,
   Krea2, Boogu, Ideogram4, Radiance, NeRF (groß/klein), Chroma/distilled
   (groß/klein), MinimaxH3, LTXv2, Gemma4 und Qwen3-VL. Die Signaturen für
   die ctq-eigenen Presets stammen direkt aus den `MODEL_FILTERS`-Konstanten
   deiner installierten `convert_to_quant`-Version. LENS bleibt bewusst ohne
   Erkennung, weil es sich zu viele Tensor-Namen mit ctqs Qwen-Dual-Stream
   teilt. Passt nichts eindeutig, bleibt die Pille weg und du wählst selbst.
6. **Voreinstellungen**: Format, Layer-Schutz und erweiterte Werte lassen
   sich unter einem Namen speichern (`presets.json`) und mit einem Klick
   wieder laden. Dateipfade gehören bewusst nicht dazu.
7. **Erweitert**: zusätzliche ctq-Argumente und der Low-Memory-Modus. Der
   steuert ctqs `--low-memory`-Flag und betrifft laut ctq-Quellcode nur den
   **System-RAM** beim Einlesen, **nicht den VRAM**. „Automatisch“ (Standard)
   schaltet es nur ein, wenn die Eingabedatei mehr als 50&nbsp;% des gerade
   verfügbaren RAM belegt (ctqs eigene Empfehlung). Gemessene Werte und
   Entscheidung stehen im Log jedes Laufs.
8. **Befehl ansehen**: zeigt den genauen `ctq`-Befehl für die aktuellen
   Einstellungen (für cmd oder PowerShell), zum Kopieren und selbst
   Ausführen. Er wird von derselben Funktion gebaut wie der echte Lauf.
9. **Quantisieren starten** (Leiste unten). Mittig öffnet sich ein Fenster
   (1000&times;500, verschieb- und vergrößerbar) mit Status und Live-Log, sein
   Rahmen läuft farbig um, solange der Job arbeitet. Fortschrittsbalken und
   Dauer stehen in der Leiste unten. Über das &times; minimierst du das Fenster,
   der Job läuft weiter und eine kleine Pille zeigt den Status.
10. **Warteschlange**: Startest du ein weiteres Modell, während eines läuft,
    reiht es sich ein. Aufträge laufen nacheinander, ein wartender lässt sich
    wieder entfernen, und über „Log“ siehst du die Ausgabe jedes Auftrags. Die
    Warteschlange liegt im Server, ein neu geladener oder geschlossener Tab
    verliert also nichts.
11. **Abbrechen**: Solange ein Job läuft, gibt es im Fenster, in der Leiste
    unten und in der Warteschlange einen „Abbrechen“-Knopf. Er beendet ctq
    samt seinem Python-Unterprozess (sonst würde die GPU weiterrechnen) und
    löscht eine unvollständige Ausgabedatei, aber nur, wenn sie vor dem Start
    noch nicht existierte.

Bei `INT8 ConvRot` und `INT8 Tensor-Wise` läuft danach automatisch der
Marker-Fix: Er kürzt die `.comfy_quant`-Marker auf genau die Felder, die
ComfyUI erwartet, damit das Ergebnis ohne die `unet unexpected`-Warnung lädt.

## Zurück nach bf16

Der Bereich „Zurück nach bf16“ wandelt ein INT8-Modell (ConvRot oder einfaches
INT8) von ctq oder ein INT6-Modell (W6A8: 6-Bit-Gewichte in Paketen mit
Gruppen-Skalen) wieder in eine normale bf16-, fp16- oder fp32-Datei um, zum
Beispiel um es mit anderen Modellen zu mergen. Du wählst das quantisierte Modell
aus der Liste (oder fügst den vollständigen Pfad ein), prüfst Ausgabename und
Datentyp und startest. Die Umwandlung läuft als Auftrag in derselben
Warteschlange (Log, Fortschritt und Abbrechen wie gewohnt).

- **Genauigkeit:** Gerechnet wird mit derselben Formel, mit der ComfyUI solche
  Dateien selbst entpackt, in fp32 ohne TF32. Auf dem Prozessor ist das Ergebnis
  bit-identisch zu dem von ComfyUI. Auf der Grafikkarte (Voreinstellung, CUDA)
  rechnen die Summen in anderer Reihenfolge, dadurch weichen etwa 0,1 % der
  Werte um eine bf16-Stufe ab, ohne an Genauigkeit zu verlieren. Reicht der freie
  Grafikspeicher nicht (z. B. weil ComfyUI läuft), rechnet das Tool die Schicht
  automatisch auf dem Prozessor. Die Dauer hängt vor allem von der Festplatte ab,
  weil die Datei gelesen und die größere bf16-Datei geschrieben werden muss. Die
  Rundung der INT8-Quantisierung (etwa 0,9 % Abweichung vom Original) lässt sich
  nicht rückgängig machen.
- **Selbstprüfung:** Jede Schicht wird nach dem Entpacken noch einmal quantisiert.
  Es müssen exakt dieselben INT8-Werte herauskommen, sonst steht eine Warnung im
  Log.
- **Unverändertes:** Alles, was nicht quantisiert wurde (Normen, Bias, geschützte
  Schichten), wird Byte für Byte kopiert. Die Hilfs-Tensoren der Quantisierung und
  die Quantisierungs-Metadaten fallen weg, die Schlüsselnamen bleiben gleich.
- **Sicherheit:** Die Datei entsteht unter dem Namen `….partial` und wird erst am
  Ende umbenannt. Eine vorhandene Datei wird nie überschrieben. Bricht man ab,
  bleibt nichts übrig.
- **Datentyp:** bf16 ist die Voreinstellung, fp16 ist etwas feiner, fp32 hat die
  höchste Genauigkeit, die Datei ist dann doppelt so groß.
- **INT6 (W6A8):** Die 6-Bit-Codes werden ausgepackt, mit den Gruppen- und
  Zeilen-Skalen zurückgerechnet und wie bei INT8 zurückgedreht. Das Ergebnis ist
  identisch mit der Rechenfunktion in `comfy_kitchen` (in fp32, ohne die
  Zwischenrundung auf bf16, die deren bf16-Weg macht). Der Verlust ist größer als
  bei INT8: etwa 2–3 % Abweichung vom Original.
- **Aufmerksamkeits-Hinweise:** Quantisierte ComfyUI-Modelle tragen manchmal pro
  Schicht einen winzigen Tensor `comfy_attention.config`. Das sind keine Gewichte,
  sie werden nicht übernommen (Merge-Werkzeuge würden über einen uint8-Tensor
  stolpern).
- **Grenzen:** Andere Formate (4-Bit-W4A8, FP8, NVFP4, MXFP8, blockweises INT8)
  werden vorerst mit einer klaren Meldung abgelehnt. Der Datentyp-Wechsel ändert
  nur die Ausgabe, nicht, was schon an Genauigkeit verloren ist.

## Logs

Jeder Quantisierungs-Lauf (voller ctq-Befehl und komplette Ausgabe) wird
dauerhaft als Datei unter `logs/` neben `quant_server.py` gespeichert und
bleibt auch nach einem Neustart erhalten. Im Abschnitt „Frühere Läufe“ kannst
du jeden Lauf nachlesen.

## Einstellungen, Updates und Systemcheck

Das Zahnrad oben rechts öffnet die Einstellungen:

- **Modell-Ordner**: beliebig viele, frei wählbar, dauerhaft gespeichert.
- **ComfyUI-Installation**: Pfad zu deinem ComfyUI-Ordner (der mit
  `custom_nodes` darin). Wird beim ersten Start anhand des
  Python-Interpreters geraten (klappt bei der üblichen
  ComfyUI-Portable-Struktur automatisch). Wird nur für den Check von
  ComfyUI-INT8-Fast gebraucht.
- **ctq-Programm**: wird normalerweise automatisch neben deinem Python
  gefunden. Nur falls das fehlschlägt, trägst du hier den Pfad zu `ctq.exe`
  ein.
- **Updates und Systemcheck** (braucht Internet):
  - **GFlava-Quant**: vergleicht die eigene Version mit dem neuesten Release
    auf GitHub. Gibt es ein neueres, führt „Release ansehen“ zur
    Download-Seite. Installiert wird dabei nichts automatisch.
  - **convert_to_quant (ctq)**: ob es installiert ist und ob PyPI eine neuere
    Version hat. „Installieren“ bzw. „Aktualisieren“ führt
    `pip install -U convert_to_quant` aus.
  - **[ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)**:
    ob das Custom-Node in `<ComfyUI>\custom_nodes\` liegt und ob es ein Update
    gibt (lokale gegen entfernte Git-Revision). „Installieren“ klont es per
    `git clone`, „Aktualisieren“ holt es per `git pull`. Das sind echte
    Git-Befehle auf deinem Rechner. Git muss dafür installiert und im `PATH`
    sein.

  Kurz nach dem Öffnen der Seite läuft diese Prüfung automatisch. Gibt es
  Updates, erscheint ein Hinweis über der Seite und eine Pille „Update
  verfügbar“ in der Kopfzeile. Beide führen zu diesem Abschnitt. Blendest du
  den Hinweis aus, kommt er erst bei einer noch neueren Version wieder, die
  Pille bleibt. Mit **Nach Updates suchen** prüfst du jederzeit von Hand neu.
  Die automatische Prüfung lässt sich mit dem Häkchen darunter abschalten
  (gilt für diesen Browser). Ihr Ergebnis wird eine Stunde lang
  wiederverwendet, damit nicht jedes Neuladen GitHub, PyPI und Git abfragt.

Modell-Ordner, ComfyUI- und ctq-Pfad liegen in `config.json` neben
`quant_server.py`. Die Datei kannst du auch von Hand bearbeiten oder sichern.

## Grenzen

- Es läuft immer nur ein Quantisierungs-Job gleichzeitig (sonst würden sich
  zwei Jobs um RAM und VRAM streiten). Weitere warten in der Warteschlange.
  Die liegt im Speicher des Servers und ist nach einem Neustart leer; die
  Logs bleiben.
- Formate außer INT8 ConvRot sind nach ctq-Dokumentation aufgerufen, aber
  nicht von uns Ende zu Ende geprüft.
- Entwickelt unter Windows. Linux und macOS laufen über `start_server.sh`: Die
  RAM-Erkennung nutzt `psutil` (oder `/proc/meminfo` bzw. `vm_stat`), Abbrechen
  beendet die ganze Prozessgruppe, `ctq` wird auch im `PATH` gesucht, und weil es
  dort keine nativen Auswahldialoge gibt, werden Ordner- und Dateipfade in einem
  Eingabefeld getippt (`~` funktioniert). Die automatischen Tests in `tests/`
  laufen über GitHub Actions unter Windows, Linux und macOS; die App selbst wurde
  bisher nur unter Windows benutzt, melde bitte Probleme auf den anderen Systemen.

## Schriftarten

Die Oberfläche nutzt [Space Grotesk](https://github.com/floriankarsten/space-grotesk),
[IBM Plex Sans](https://github.com/IBM/plex) und
[JetBrains Mono](https://github.com/JetBrains/JetBrainsMono). Sie liegen unter
`fonts/` und werden lokal ausgeliefert, die Seite lädt nichts aus dem
Internet. Alle drei stehen unter der SIL Open Font License 1.1, siehe
`fonts/OFL.txt`.
