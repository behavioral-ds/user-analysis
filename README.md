User Analysis on Auspol
================

## Tools

``` r
library(evently)
library(reticulate)
birdspotter <- import('birdspotter')
```

## Dataset

The `auspol` dataset contains 9534 tweets curated by querying the
Twitter steaming API for tweet containing the hashtag `#auspol` (short
for Australian politics) during ????. Due to the terms of service of
Twitter, we only publish the tweet IDs of these tweets. One can recover
the complete dataset in JSONL via a tool named
[twarc](https://github.com/DocNow/twarc) by the following bash command

``` bash
twarc hydrate auspol-ids.txt > tweets.jsonl
```

## Extracting diffusion cascades

``` r
cascades <- parse_raw_tweets_to_cascades('auspol.jsonl', keep_user = T)
```

``` r
bs <- birdspotter$BirdSpotter('auspol.jsonl')
labeledUsers <- bs$getLabeledUsers(out = './output.csv')
```

## User influence analysis

> TODO

## User botness prediction

> TODO
