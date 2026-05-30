import os
from google import genai
from google.genai import types

def translate_code_switched_text(raw_text: str, primary_local_language: str = "Hindi") -> str:
    """
    Translates raw code-switched text (e.g. Hinglish) to pure English
    using the Gemini 1.5 Flash model and modern google-genai SDK.
    """
    # The client automatically picks up the GEMINI_API_KEY environment variable.
    client = genai.Client()
    
    system_instruction = f"""You are an expert Contextual Translation Engine specializing in Code-Switching NLP.
Your task is to take a raw text string that mixes English and {primary_local_language} and translate the entire meaning into pure, fluent, contextually accurate English.

STRICT INSTRUCTIONS:
1. Detect all mixed languages present.
2. Translate everything into fluent English.
3. CRITICAL: Watch out for "Cross-Lingual False Friends" (homophones). For example, the English word "come" vs. the Hindi word "kam" (meaning "less" or "work"). You must use the surrounding sentence syntax to disambiguate the true meaning.
4. Output ONLY the final English string. No conversational filler, no explanations, no markdown formatting (like ```).
"""

    # Pass the system instruction using the modern config syntax
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=raw_text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2, # Keep temperature low for accurate, factual translation
        )
    )
    
    return response.text.strip()

if __name__ == "__main__":
    # Simple check for the API key to assist the developer
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set.")
        print("The modern google-genai SDK requires this to be set in your environment.")
        print("Please set it in your terminal before running this test:")
        print("  Windows: set GEMINI_API_KEY=your_api_key_here")
        print("  Unix:    export GEMINI_API_KEY=your_api_key_here")
        print("\nSkipping live test due to missing key.")
    else:
        print("Testing Code-Switching Contextual Translation Engine...")
        mock_hinglish = "Mujhe kal market jana hai to buy some apples"
        print(f"Raw Input:  {mock_hinglish}")
        
        try:
            english_translation = translate_code_switched_text(mock_hinglish, "Hindi")
            print(f"Translation: {english_translation}")
            print("\nTest completed successfully!")
        except Exception as e:
            print(f"Error during translation API call: {e}")
