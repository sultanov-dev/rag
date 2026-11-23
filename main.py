from rag import build_index, build_prompt, retrieve
from utils import load_text


def main() -> None:
    text = load_text("data.txt")
    chunks, idf, vectors = build_index(text, chunk_size=18, overlap=2)

    while True:
        query = input("\nSavol (chiqish uchun Enter): ").strip()

        if not query:
            break

        retrieved = retrieve(query, chunks, idf, vectors)

        if not retrieved:
            print("Mos ma'lumot topilmadi")
            continue

        prompt = build_prompt(query, retrieved)

        print(prompt)


if __name__ == "__main__":
    main()
