import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.file_manager import load_file, save_file
from config import config

class TextToGloss:
    PROMPT = """You are an expert ASL linguist. Convert English to ASL gloss following these strict rules:


1. ASL SYNTAX: Use topic-comment structure. Time indicators first, then subject, object, verb.
2. CLASSIFIERS: Use CL: for classifiers (CL:1=person, CL:3=vehicle, etc.)
3. NON-MANUAL MARKERS: Do NOT include (e.g., "mm", "cs", "hs", "rb").
4. COMPOUNDS: Do NOT hyphenate multi-word concepts (e.g., GO TO instead of GO-TO).
5. TENSE: Shown through time indicators (PAST, FUTURE) or context.
6. PRONOUNS: Exclude IX references. Use the closest direct reference instead.
7. SPATIAL GRAMMAR: Establish locations descriptively without IX-loc references.
8. ROLE SHIFT: Indicate with RS: when changing perspectives.
9. FINGERSPELLING: Use # for lexicalized signs (e.g., #JOB).


Handle these special cases:
- Questions: Convert to statement form without "?".
- Exclamations: Remove "!".
- Negation: Do not use "hs" marker; instead, use explicit negation words.
- Imperatives: Use strong directional verbs.
- Conditionals: Do not use "rb"; rephrase for clarity.


Provide only the ASL gloss, no explanations, no punctuation. Use uppercase for signs.
"""
    
    def __init__(self):
        self.input_file = config.OUTPUT_DIR / "transcription.txt"
        self.output_file = config.OUTPUT_DIR / "gloss.txt"
        
    def run(self):
        try:
            text = load_file(self.input_file)
            gloss = self._convert_to_gloss(text)
            save_file(self.output_file, gloss)
            return True
        except Exception as e:
            print(f"Error in text-to-gloss: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _convert_to_gloss(self, text):
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.PROMPT},
                {"role": "user", "content": f"TRANSLATE: {text}"}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()