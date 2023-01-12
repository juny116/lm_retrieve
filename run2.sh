datasets="trec-covid dbpedia-entity fever quora arguana nq"

for dataset in $datasets; do
python generate_syn.py task=$dataset templates=$dataset generator=flan3b_beam
done
