# Label Policy

## Class IDs

- `0` = `arabic_moroccan`
- `1` = `arabizi_moroccan`
- `2` = `other`

## Row Schema

The final dataset must contain these columns:

- `text`: normalized text content
- `dialect`: normalized dialect name
- `src`: source dataset identifier
- `has_latin_script`: `true` if the row contains at least one Latin character, otherwise `false`
- `label`: class ID (`0`, `1`, or `2`)

## Labeling Rules

- Use label `0` for Moroccan Darija written mostly in Arabic script.
- Use label `1` for Moroccan Darija written mostly in Latin/Arabizi script.
- Use label `2` for everything else:
  - non-Moroccan Arabic dialects
  - MSA
  - English or French translations
  - any other non-Darija text

## Mixed-Script / Mixed-Language Rows

- If a Moroccan Darija row contains a few English or French words but remains mostly Moroccan Darija in Arabic script, keep it as label `0`.
- If a Moroccan Darija row is mainly written in Latin/Arabizi, use label `1`.
- The `has_latin_script` flag is independent from the class label.
  - A row can be label `0` and still have `has_latin_script = true` if it contains a few Latin tokens.

## Deduplication

- If the same normalized text appears several times, it should exist only once in the final dataset.
- Source provenance should still be preserved in `src`.
