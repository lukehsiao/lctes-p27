#!/usr/bin/env bash

# Specifically meant to run on Raiders
dropdb --if-exists transistor_scale_rels
createdb transistor_scale_rels

# Only parse the first time, no need to reparse the same docs over and over.
transistors --log-dir transistor_scale_rels_logs --gpu 0 --conn-string "postgresql:///transistor_scale_rels" --parallel=8 --max-docs=1000 --parse --first-time --stg-temp-min 
# Truncate, which is faster and more similar to a fresh DB than dropping via query.
psql -d transistor_scale_rels -c "truncate mention cascade;"
psql -d transistor_scale_rels -c "truncate candidate cascade;"
psql -d transistor_scale_rels -c "truncate label_key cascade;"
psql -d transistor_scale_rels -c "truncate feature_key cascade;"
psql -d transistor_scale_rels -c "vacuum analyze;"


transistors --log-dir transistor_scale_rels_logs --gpu 0 --conn-string "postgresql:///transistor_scale_rels" --parallel=8 --max-docs=1000 --first-time --stg-temp-min --stg-temp-max
psql -d transistor_scale_rels -c "truncate mention cascade;"
psql -d transistor_scale_rels -c "truncate candidate cascade;"
psql -d transistor_scale_rels -c "truncate label_key cascade;"
psql -d transistor_scale_rels -c "truncate feature_key cascade;"
psql -d transistor_scale_rels -c "vacuum analyze;"

transistors --log-dir transistor_scale_rels_logs --gpu 0 --conn-string "postgresql:///transistor_scale_rels" --parallel=8 --max-docs=1000 --first-time --stg-temp-min --stg-temp-max --polarity
psql -d transistor_scale_rels -c "truncate mention cascade;"
psql -d transistor_scale_rels -c "truncate candidate cascade;"
psql -d transistor_scale_rels -c "truncate label_key cascade;"
psql -d transistor_scale_rels -c "truncate feature_key cascade;"
psql -d transistor_scale_rels -c "vacuum analyze;"

transistors --log-dir transistor_scale_rels_logs --gpu 0 --conn-string "postgresql:///transistor_scale_rels" --parallel=8 --max-docs=1000 --first-time --stg-temp-min --stg-temp-max --polarity --ce-v-max
