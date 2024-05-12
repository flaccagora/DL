# Deep_Learning

# Train Dataset
In leonardo_scratch/large/userexternal/mnunzian trovate la cartella data.
per Noemi work in progress, per il momento basti sapere che lavoreremo con file di testo

# TODO
(consiglio lettura dello script di train per capire problematiche e come muovesi.. )

- Vocabulary
- data.arrow -> data.txt (using hf load_dataset dovrebbe funzionare)
- pretrained Tokenizer  [mi pare di capire che il tokenizer presente sia una sorta di dummy, dovremmo lavorare con 
qualcosa del tipo tokenizer = AutoTokenizer.from_pretrained("gpt2") e quindi capire come inserirlo nella pipeline]


# Varie ed eventuali
nei prossimi giorni provo a far partire un train con il dataset di test, per vedere se ci sono o meno problemi di diversa natura, per poi 
implementare la vera pipeline (in costruzione, vedi TODO).