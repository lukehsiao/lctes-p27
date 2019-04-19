#!/usr/bin/env bash

# Specifically meant to run on Raiders
for DOCS in 200 400 800 1600 3200
do
  dropdb --if-exists transistor_scale_docs
  createdb transistor_scale_docs
  echo "$DOCS Documents..."
  transistors --log-dir transistor_scale_docs_logs --gpu 0 --conn-string "postgresql:///transistor_scale_docs" --parallel=8 --parse --first-time --stg-temp-min --max-docs $DOCS
done
