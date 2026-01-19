class Prompts:
    @staticmethod
    def get_no_guidance_prompt():
        return """Task: Determine the author ID of the query text from the given potential authors.
Mandatory Rules:
1. Must select an author ID from the provided potential authors (do not return "Unsure" or "cannot determine").
2. If uncertain, guess based on the most obvious style feature.
3. Only output the STRICT JSON object as required. Do NOT add any extra text, comments, explanations, or line breaks. JSON must be valid and complete.
"""

    @staticmethod
    def get_little_guidance_prompt():
        return """Task: Determine the author ID of the query text from the given potential authors.
Mandatory Rules:
1. Must select an author ID from the provided potential authors (do not return "Unsure" or "cannot determine").
2. Do not consider topic differences.
3. If uncertain, guess based on the most obvious style feature.
4. Only output the STRICT JSON object as required. Do NOT add any extra text, comments, explanations, or line breaks. JSON must be valid and complete.
"""

    @staticmethod
    def get_grammar_prompt():
        return """Task: Determine the author ID of the query text from the given potential authors.
Mandatory Rules:
1. Must select an author ID from the provided potential authors (do not return "Unsure" or "cannot determine").
2. Focus on grammatical styles.
3. If uncertain, guess based on the most obvious style feature.
4. Only output the STRICT JSON object as required. Do NOT add any extra text, comments, explanations, or line breaks. JSON must be valid and complete."""

    @staticmethod
    def get_lip_prompt():
        return """Task: Determine the author ID of the query text from the given potential authors.
Mandatory Rules:
1. Must select an author ID from the provided potential authors (do not return "Unsure" or "cannot determine").
2. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Focus on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings.
3. If uncertain, guess based on the most obvious style feature.
4. Only output the STRICT JSON object as required. Do NOT add any extra text, comments, explanations, or line breaks. JSON must be valid and complete."""

    @staticmethod
    def get_system_message():
        return """Respond with a JSON object including two key elements:
{
  "analysis": Reasoning behind your answer.
  "answer": The query text's author ID.
}"""