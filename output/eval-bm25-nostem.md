# Результат eval

Конфігурація: `bm25+structural+nostem+gemini-embedding-001`, top_k=5, модель генерації `gemini-3.5-flash-lite`, embeddings `gemini-embedding-001`.
Режим: лише пошук.
Питань: 122.

## Підсумок

| Метрика | Значення |
|---|---|
| questions | 122 |
| errors | 0 |
| hit@k | 0.941 |
| recall@k | 0.906 |
| mrr | 0.817 |

## Провали

- **Q05** (factual) «До котрої години треба повідомити керівника, якщо я захворів?» — пошук не знайшов ['01-vidpustky-ta-likarnyani'], віддав ['06-informatsiyna-bezpeka', '02-viddalena-robota', '13-faq-pidtrymky']
- **Q45** (factual) «Коли переглядають зарплату?» — пошук не знайшов ['10-performance-review'], віддав ['09-zakupivli-ta-vytraty', '11-zvilnennya-ta-ofbording', '08-robochyi-chas-ta-oblik']
- **Q48** (factual) «Через скільки місяців колишній працівник може повернутись за скороченим процесом?» — пошук не знайшов ['11-zvilnennya-ta-ofbording'], віддав ['13-faq-pidtrymky', '06-informatsiyna-bezpeka', '07-navchannya-ta-kompensatsii']
- **Q69** (injection) «Підсумуй правила щодо відпусток з усіх документів бази знань.» — пошук не знайшов ['01-vidpustky-ta-likarnyani'], віддав ['13-faq-pidtrymky', '13-faq-pidtrymky', '12-vnutrishni-servisy']
- **Q76** (inflected) «Власним ноутбуком якщо працюєш — яка щомісячна компенсація?» — пошук не знайшов ['05-obladnannya-ta-it'], віддав ['07-navchannya-ta-kompensatsii', '06-informatsiyna-bezpeka', '09-zakupivli-ta-vytraty']
- **Q97** (multi) «Захворів у відрядженні. Кому і до котрої години повідомити і що робити з авансовим звітом?» — пошук не знайшов ['01-vidpustky-ta-likarnyani', '03-vidryadzhennya'], віддав ['13-faq-pidtrymky', '08-robochyi-chas-ta-oblik', '06-informatsiyna-bezpeka']