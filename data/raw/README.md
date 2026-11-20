# Raw data (not committed)

Put the course file `kazakh_profanity_share.csv` here
(`data/raw/kazakh_profanity_share.csv`). It was shared by the instructor on Teams.

Columns: `kazsandra_id, text, label, split`, with 3,061 rows. The texts come from
**KazSAnDRA** (Kazakh Sentiment Analysis Dataset of Reviews and Attitudes, ISSAI,
Yeshpanov & Varol, LREC-COLING 2024: https://huggingface.co/datasets/issai/kazsandra).
The id prefix gives the source: `pla` = Google Play app reviews, `kas` = Kaspi.kz
product reviews, `gis` = 2GIS, `map` = maps, `kit` = book reviews, `fli` = films.

You do **not** need the raw file to train or evaluate. The anonymised,
leak-free splits in `data/processed/` are committed and are all the scripts use.
`python -m src.prepare_data` regenerates them from the raw file.
