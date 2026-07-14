from collections import Counter, defaultdict, deque
from pathlib import Path

input_path = Path("src/harshu_ai_os/kernel/runtime_input.txt")
word_list = input_path.read_text().lower().replace("_", " ").replace("=", " ").split()
word_count = Counter(word_list)
common_words = word_count.most_common(3)
print(common_words)

grouped_words = defaultdict(list)

for word in word_list:
    first_letter = word[0]
    grouped_words[first_letter].append(word)

print(grouped_words)

recent_words = deque(maxlen=3)

for word in word_list:
    recent_words.append(word)

print(recent_words)