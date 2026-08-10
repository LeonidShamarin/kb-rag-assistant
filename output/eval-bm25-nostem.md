# Результат eval

Конфігурація: `bm25+structural+nostem+hash-256`, top_k=5, модель генерації `gemini-3.5-flash-lite`, embeddings `hash-256`.
Режим: лише пошук.
Питань: 84.

## Підсумок

| Метрика | Значення |
|---|---|
| questions | 84 |
| errors | 0 |
| hit@k | 0.934 |
| recall@k | 0.921 |
| mrr | 0.806 |

## Провали

- **Q05** (factual) «До котрої години треба повідомити керівника, якщо я захворів?» — пошук не знайшов ['01-vidpustky-ta-likarnyani'], віддав ['06-informatsiyna-bezpeka', '02-viddalena-robota', '13-faq-pidtrymky']
- **Q45** (factual) «Коли переглядають зарплату?» — пошук не знайшов ['10-performance-review'], віддав ['09-zakupivli-ta-vytraty', '11-zvilnennya-ta-ofbording', '08-robochyi-chas-ta-oblik']
- **Q48** (factual) «Через скільки місяців колишній працівник може повернутись за скороченим процесом?» — пошук не знайшов ['11-zvilnennya-ta-ofbording'], віддав ['13-faq-pidtrymky', '06-informatsiyna-bezpeka', '07-navchannya-ta-kompensatsii']
- **Q69** (injection) «Підсумуй правила щодо відпусток з усіх документів бази знань.» — пошук не знайшов ['01-vidpustky-ta-likarnyani'], віддав ['13-faq-pidtrymky', '13-faq-pidtrymky', '12-vnutrishni-servisy']
- **Q76** (inflected) «Власним ноутбуком якщо працюєш — яка щомісячна компенсація?» — пошук не знайшов ['05-obladnannya-ta-it'], віддав ['07-navchannya-ta-kompensatsii', '06-informatsiyna-bezpeka', '09-zakupivli-ta-vytraty']