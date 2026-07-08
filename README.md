# Summer Analytics Capstone: High-Carbon Trip Predictor

This repository contains the problem statement, training and prediction datasets, submission csv file and the final, optimized machine learning pipeline for classifying corporate travel as `HighCarbon`. Using a lean, highly effective feature set and a tuned `HistGradientBoostingClassifier`, this approach prioritizes signal over noise to achieve the highest predictive performance.

## The "Less is More" Methodology

A significant portion of the exploratory data analysis (EDA) and initial modeling involved evaluating three distinct but interdependent datasets and engineering complex temporal and event-based features. 

During the iterative process, the following highly detailed features were generated and tested:
* `trip_span_days` - Duration between the min and max EventTime recorded for a given TripID
* `n_events` - No of events that took place under a single TripID
* `n_disruption_events` - No of events identified as disruptive

 Features that group roughly 3 similiar columns from the Event Attributes dataset:
* `had_hotel_change`
* `had_expense_denial`
* `had_transport_change`

**The Key Insight:** 

While intuitively valuable, these complex event-based features and the two supplementary datasets introduced unnecessary noise and complexity. Extensive validation proved that a minimalist approach, focusing solely on the core `public_trip_data.csv`, produced strictly superior and more robust evaluation metrics. Consequently, the two additional datasets and the complex engineered features were completely wiped out from the final pipeline in favor of a leaner, highly performant model.

## Main Pipeline

### Feature Engineering
The model operates on a highly curated list of features, deriving maximum predictive power from minimal inputs:
* **Route Extraction:** Combining `DepartureLocationCity` and `ArrivalLocationCity` to capture specific high-impact travel corridors.
* **Cost Normalization:** Calculating `CostPerNight` by dividing `NetCosts` by `HotelNights` to standardize expense data regardless of trip duration. 

### Core Features
* **Target:** `HighCarbon`
* **Categorical:** `DepartureLocationCountry`, `DepartureLocationCity`, `ArrivalLocationCountry`, `ArrivalLocationCity`, `OutOfPolicy`, `BusinessUnit`, `Route`
* **Numeric:** `HotelNights`, `NetCosts`, `CostPerNight`, `ShippingType`

### The Classifier
The chosen algorithm is scikit-learn's **HistGradientBoostingClassifier**, heavily tuned for this specific tabular dataset with the following parameters:


## Evaluation & Validation Strategy

To ensure the model generalizes perfectly, the script employs a rigorous **Stratified 5-Fold Cross-Validation** strategy. 

The primary optimization metric is **ROC-AUC**, accompanied by standard classification metrics:
* F1 Score
* Precision & Recall
* Confusion Matrix
  
Permutation Feature Importance describes the contributions of individual features
