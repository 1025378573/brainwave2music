from gpt.chatgpt import generate_text_from_keywords
from text_to_music import generate_music_from_text
from melody_to_music import generate_music_from_melody

def get_keywords():
    keywords = input("Please enter keywords separated by spaces: ")
    return keywords.split()


def choose_mode():
    """
    Prompt the user to choose between text mode or melody mode.
    Returns:
        str: 'text' or 'melody' based on user input.
    """
    while True:
        mode = input("Choose mode: 'text' for text-to-music or 'melody' for melody-to-music: ").strip().lower()
        if mode in ['text', 'melody']:
            return mode
        print("Invalid input. Please enter 'text' or 'melody'.")

def main():
    mode = choose_mode()

    if mode == 'text':
        # Text-to-Music Mode
        keywords = get_keywords()
        description = generate_text_from_keywords(keywords)
        if not description:
            print("Failed to generate text description. Exiting.")
            return
        print(f"Generated text description: {description}")
        generate_music_from_text(description)
    
    elif mode == 'melody':
        # Melody-to-Music Mode
        n_segments = int(input("Enter the number of EEG segments to process: "))

        keywords = get_keywords()
        description = generate_text_from_keywords(keywords)
        if not description:
            print("Failed to generate text description. Exiting.")
            return

        print(f"Processing {n_segments} segments with description: {description}")
        generate_music_from_melody(description, n_segments)

if __name__ == "__main__":
    main()
