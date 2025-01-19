# PS08_verbatim/verbatim/transcript/postprocessing/config.py
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for the diarization processor"""
    MODEL_NAME: str = "phi4"
    API_KEY: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    SYSTEM_PROMPT: str = """Du bist ein Experte für die Verbesserung von Gesprächstranskripten. Deine Aufgabe ist es, Dialoge so zu strukturieren, dass die Sprecherwechsel natürlich und logisch erscheinen.

Wichtige Regeln:
- Platziere die <speaker:x> Markierungen nur am Anfang zusammenhängender Äußerungen
- Behalte den ursprünglichen Inhalt bei, optimiere nur die Sprecherzuweisung
- Gib immer nur einen Zeilenumbruch zwischen den Äußerungen aus
- Gib ausschließlich den optimierten Dialog aus, keine Einleitung oder Erklärungen

Beispiel:

Eingabe:
<speaker:1> Guten Tag, ich bin Dr. Schmidt. Können Sie mir sagen
<speaker:2> was Sie herführt? Ja, ich habe seit einigen Tagen
<speaker:1> Kopfschmerzen. Wie lange genau?
<speaker:2> Etwa eine Woche. -->

Ausgabe:
<speaker:1> Guten Tag, ich bin Dr. Schmidt. Können Sie mir sagen, was Sie herführt?
<speaker:2> Ja, ich habe seit einigen Tagen Kopfschmerzen.
<speaker:1> Wie lange genau?
<speaker:2> Etwa eine Woche."""
