User Analysis on Auspol
================

## Tools

``` r
library(evently)
library(reticulate)
birdspotter <- import('birdspotter')
```

## Dataset

> TODO: what is auspol about

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
