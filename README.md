User Analysis on Auspol
================

## Tools

In this tutorial, we apply two tools for analyzing Twitter users,
`BirdSpotter` and `evently`. While `BirdSpotter` captures the social
influence and botness of Twitter users, `evently` specifically models
the temporal dynamics of online information diffusion. We leverage
information provided by the tools to study the users in a dataset
relating to Australian politics.

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
twarc hydrate auspol-ids.txt > auspol.jsonl
```

We note that 7077 tweets are available on Twitter as of 26/05/2020.

## Extracting diffusion cascades

At this step, we seek to extract diffusion cascades from the `auspol`
dataset for analyzing user influence and botness. A diffusion cascade
consist of a initial tweet posted by a Twitter user and followed then by
a sereis of retweets. The

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
