import ctranslate2
import transformers
from huggingface_hub import snapshot_download

# 1. Download the CTranslate2 model from the specific repo you found
repo_id = "Adeptschneider/nllb-200-distilled-600M-int8"
print(f"Checking/Downloading model from {repo_id}...")
# This safely downloads the model and returns the local folder path
model_path = snapshot_download(
    repo_id=repo_id,
    local_dir="./models/nllb",
    local_dir_use_symlinks=False # Crucial for Windows so PyInstaller doesn't break!
)

# 2. Load the AI Engine (CTranslate2) into the CPU
print("Loading translation engine into CPU...")
translator = ctranslate2.Translator(model_path, device="cpu")

# 3. Load the Dictionary (Tokenizer)
# We pull the tokenizer from the original Facebook model, as the vocabulary is exactly the same.
print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(text, source_lang="eng_Latn", target_lang="hin_Deva"):
    """
    Translates text using NLLB. 
    Requires specific FLORES-200 language codes (e.g., eng_Latn, hin_Deva).
    """
    # Tell the tokenizer what language we are starting with
    tokenizer.src_lang = source_lang
    
    # Break the text down into AI tokens
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    
    # Run the fast C++ translation engine
    # We pass 'target_prefix' to force the AI to output in our desired language
    results = translator.translate_batch(
        [source_tokens], 
        target_prefix=[[target_lang]]
    )
    
    # Grab the output tokens (ignoring the first token, which is just the language tag)
    target_tokens = results[0].hypotheses[0][1:]
    
    # Decode the tokens back into human-readable text
    translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
    
    return translated_text

# --- Test the Pipeline ---
if __name__ == "__main__":
    print("\n" + "="*50)
    
    # Test 1: English to Hindi (Fixing the ALL CAPS issue we had earlier!)
    english_input = "THE MEN GAVE A SHOUT OF TRIUMPH."
    normalized_english = english_input.capitalize() # NLLB prefers normal casing
    
    hindi_output = translate(normalized_english, source_lang="eng_Latn", target_lang="hin_Deva")
    print(f"English: {normalized_english}")
    print(f"Hindi:   {hindi_output}")
    
    print("-" * 50)
    
    # Test 2: Hindi back to English
    hindi_input = "नमस्ते, आप कैसे हैं?" # "Hello, how are you?"
    english_output = translate(hindi_input, source_lang="hin_Deva", target_lang="eng_Latn")
    print(f"Hindi:   {hindi_input}")
    print(f"English: {english_output}")
    
    print("="*50 + "\n")