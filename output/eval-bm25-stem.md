# Результат eval

Конфігурація: `bm25+structural+stem+gemini-embedding-001`, top_k=5, модель генерації `gemini-3.5-flash-lite`, embeddings `gemini-embedding-001`.
Режим: лише пошук.
Питань: 122.

## Підсумок

| Метрика | Значення |
|---|---|
| questions | 122 |
| errors | 0 |
| hit@k | 0.98 |
| recall@k | 0.955 |
| mrr | 0.91 |

## Провали

- **Q05** (factual) «До котрої години треба повідомити керівника, якщо я захворів?» — пошук не знайшов ['01-vidpustky-ta-likarnyani'], віддав ['06-informatsiyna-bezpeka', '02-viddalena-robota', '13-faq-pidtrymky']
- **Q45** (factual) «Коли переглядають зарплату?» — пошук не знайшов ['10-performance-review'], віддав ['09-zakupivli-ta-vytraty', '11-zvilnennya-ta-ofbording', '02-viddalena-robota']