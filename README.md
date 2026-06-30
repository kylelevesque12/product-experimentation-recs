# Product Experimentation Recommender

Should a product team launch a new personalized recommendation algorithm?

This project is a product data science case study inspired by Netflix's recommendation system. Netflix's own engineering writing emphasizes that recommendation quality is not only about predicting ratings; production recommenders also balance popularity, personalization, ranking, diversity, catalog discovery, and online experimentation. This project applies that product mindset to a smaller, reproducible MovieLens analysis.

The goal is not to completely recreate and implement Netflix's entire recommendation system, as I do not currently have access to a large scale system where I can deploy and actually test recommendations, but to explore and learn more about the beginning stages of this process. More specifically, I aim to answer the question:

> Given historical user ratings, is a personalized hybrid recommender strong enough to move forward to an online A/B test?

## Evaluation Framing

This project uses several recommendation metrics, but they do not all play the same role in the final product decision.

The main decision metrics are:

- **Rated Precision@10:** among recommended movies that a user later rated, what share were positive?
- **Catalog Coverage:** how much of the movie catalog appears in recommendations across users?

These are emphasized because MovieLens does not include impression logs. In other words, we do not know which movies users actually saw or had the opportunity to rate. A missing rating does not mean the user disliked the movie; it may simply mean the user was never exposed to it.

Standard ranking metrics are still reported as secondary diagnostics:

- **Precision@10:** how many top-10 recommendations appear in the user's future positive ratings?
- **Recall@10:** how much of the user's future positive ratings were recovered by the recommendations?
- **NDCG@10:** did the recommender rank future positive ratings near the top?

These metrics are useful, but in this offline setting they tend to favor popular movies because popular movies are more likely to have been watched and rated historically. For that reason, the final recommendation prioritizes whether the hybrid improves discovery and maintains positive observed response, rather than simply maximizing standard offline ranking metrics.

## Project Summary

I built and evaluated several movie recommendation policies:

- Popularity baseline
- Genre-personalized recommender
- Content-based recommender using TMDb movie descriptions
- Hybrid recommenders combining popularity, content similarity, and genre preference

The final selected challenger is a three-signal hybrid:

```text
50% popularity
30% content similarity
20% genre preference
```

The content representation uses:

```text
TMDb overview + MovieLens genres + TMDb genres
```

## Recommendation Policies

The project compares several recommendation strategies, starting with simple baselines and gradually adding personalization.

### Popularity Baseline

The popularity baseline recommends movies with the most positive ratings in the training data, excluding movies the user has already rated. This is a strong baseline because popular movies are often broadly liked, but it tends to recommend the same small set of movies to many users.

### Genre-Personalized Recommender

The genre recommender builds a simple user profile from genres the user rated positively in the past. It then recommends unseen movies that match those genre preferences. This adds interpretability, but genres are broad and can over-simplify user taste.

### TF-IDF Content Recommender

The content-based recommender uses TMDb movie descriptions and genre text to represent each movie. I used TF-IDF to convert text into numerical features, then built a user taste profile by averaging the vectors of movies the user rated positively.

The intuition is:

```text
If a user liked movies with similar descriptions, themes, and genres,
recommend unseen movies with similar content.
```

I tested several content representations:

- overview only
- overview + tagline
- overview + genres
- title + genres + overview + tagline

The strongest content representation was:

```text
overview + MovieLens genres + TMDb genres
```

This suggested that plot descriptions and genre information added useful semantic signal, while titles and taglines did not clearly improve performance.

### Hybrid Recommenders

The final models combine multiple signals:

- global popularity
- user-specific content similarity
- user-specific genre preference

The selected challenger uses:

```text
50% popularity
30% content similarity
20% genre preference
```

This model was chosen because it meaningfully increases catalog coverage while maintaining competitive rated-only Precision@10. Leaning too far into popularity significantly decreases catalog coverage, while leaning too far into content similarity and genre preference significantly decreases rated-only Precision@10. Therefore, a more balanced mix between the three signals tends to be a better tradeoff for the goals of this recommendation system.

## Final Recommendation

The hybrid recommender is a strong candidate for an online A/B test.

I would not recommend launching it directly from offline results alone, because MovieLens does not include impression logs and the rated-precision lift is not statistically conclusive in this offline setting. However, the treatment model substantially improves catalog coverage and has slightly higher rated Precision@10 than the popularity baseline.

That makes it a credible product candidate for the next stage:

```text
Move the selected hybrid recommender into an online experiment,
where the team can measure real user exposure, clicks, viewing behavior,
retention, and satisfaction.
```

The main tradeoff is that standard offline ranking metrics are lower than the popularity baseline. I treat those metrics as secondary diagnostics because they are biased by historical exposure: popular movies were more likely to have been watched and rated in the first place. In a more controlled setting where impression logs are included, these metrics would be more useful, but given the limitations with this analysis, they are not the priority here. 

## Key Findings

Control policy:

```text
Popularity recommender
```

Treatment policy:

```text
Hybrid recommender: 50% popularity, 30% content, 20% genre
```

| Metric | Popularity Control | Hybrid Treatment |
|---|---:|---:|
| Rated Precision@10 | 0.7719 | 0.7802 |
| Catalog Coverage | 0.93% | 5.21% |
| Unique Recommended Movies | 91 | 508 |
| Avg Global Positive Ratings | 165.1 | 92.7 |
| Precision@10 | 0.0579 | 0.0425 |
| Recall@10 | 0.0497 | 0.0359 |
| NDCG@10 | 0.0759 | 0.0535 |

Paired user-level comparison for rated Precision@10:

```text
Mean treatment-control difference: +0.0409
95% CI: -0.0188 to +0.1006
```

Interpretation:

- The hybrid recommends a much broader set of movies.
- Rated-only Precision@10 is slightly higher for the hybrid.
- The confidence interval crosses zero, so the rated-precision lift is uncertain.
- Standard ranking metrics favor popularity, likely because historical ratings are exposure-biased toward popular movies.

## Why Catalog Coverage Matters

Popularity is a strong recommender baseline because many users like broadly popular movies. But it can also recommend the same small set of movies to everyone.

For a product team, that creates a tradeoff:

```text
Higher short-term offline accuracy
vs.
more personalized discovery and broader catalog exposure
```

In this project, catalog coverage is treated as a primary product metric because the business question is not only "Can we predict known future ratings?" but also "Can personalization help users discover a wider set of relevant movies?"

## Data

This project uses MovieLens `latest-small` from GroupLens:

- https://grouplens.org/datasets/movielens/latest/
- https://doi.org/10.1145/2827872

MovieLens provides:

- User ratings
- Movie titles
- Movie genres
- Links to external IDs such as TMDb IDs

This project also uses TMDb metadata, fetched through the TMDb API, to add:

- Movie overviews
- Taglines
- TMDb genres
- TMDb popularity and vote metadata

TMDb metadata is used only as supplemental project data. If using or extending this project, review and follow the [TMDb API Terms of Use](https://www.themoviedb.org/documentation/api/terms-of-use) and [TMDb attribution requirements](https://www.themoviedb.org/about/logos-attribution). This project does not commit TMDb metadata to Git and is not endorsed or certified by TMDb.

Raw and processed data are intentionally not committed to Git.

## Project Structure

```text
product-experimentation-recs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_recommendations.ipynb
│   ├── 03_genre_personalization.ipynb
│   ├── 04_content_feature_evaluation.ipynb
│   ├── 05_hybrid_model_selection.ipynb
│   └── 06_offline_policy_comparison.ipynb
├── scripts/
├── src/
└── tests/
```

## Notebook Roadmap

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Explore MovieLens users, movies, ratings, sparsity, and genres |
| `02_baseline_recommendations.ipynb` | Build and evaluate the popularity baseline |
| `03_genre_personalization.ipynb` | Compare genre, popularity, content, and hybrid recommenders |
| `04_content_feature_evaluation.ipynb` | Inspect and compare content text features |
| `05_hybrid_model_selection.ipynb` | Tune hybrid model weights and select the challenger |
| `06_offline_policy_comparison.ipynb` | Compare control vs treatment with paired user-level uncertainty |

## Methods

### Data Preparation

- Download MovieLens latest-small
- Clean ratings
- Convert Unix timestamps to datetimes
- Define positive interactions as ratings `>= 4.0`
- Create a leakage-safe temporal user split

### Recommenders

- Popularity recommender
- Genre-personalized recommender
- TF-IDF content recommender
- Popularity + content hybrid
- Popularity + content + genre hybrid

### Evaluation

Accuracy diagnostics:

- Precision@10
- Recall@10
- NDCG@10

Product diagnostics:

- Rated Precision@10
- Catalog coverage
- Unique recommended movies
- Average global popularity of recommended movies

Policy comparison:

- Control: popularity baseline
- Treatment: selected hybrid recommender
- Paired user-level treatment-control differences
- 95% confidence intervals

## Reproducing the Project

Create and activate a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

To fetch TMDb metadata, set a TMDb API key in your terminal:

```bash
export TMDB_API_KEY="your_key_here"
python -m scripts.fetch_tmdb_metadata
```

Do not commit API keys, raw data, or processed data.

## Limitations

This project is intentionally honest about what offline recommender evaluation can and cannot show.

Main limitations:

- No real online A/B test
- No impression logs
- No direct observation of which movies users were exposed to
- Missing ratings do not imply dislikes
- MovieLens behavior may not reflect a modern streaming product
- TMDb metadata improves item descriptions but does not replace true behavioral logs

The conclusion is therefore:

```text
The hybrid recommender is not proven to cause better engagement.
It is a credible candidate for an online test.
```

## Inspiration

Netflix's recommendation work motivated the framing of this project: recommender systems are product systems, not just modeling exercises. Netflix has written about moving beyond rating prediction toward ranking, personalization, similarity, popularity, diversity, and online experimentation.

Useful references:

- Netflix TechBlog: [Netflix Recommendations: Beyond the 5 stars, Part 1](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
- Netflix TechBlog: [Netflix Recommendations: Beyond the 5 stars, Part 2](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5)
- Gomez-Uribe and Hunt, [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://doi.org/10.1145/2843948)
- Harper and Konstan, [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872)
