# Summer Analytics Capstone

This repository contains the problem statement, final presentation, training and prediction datasets, submission csv file and the final optimized machine learning pipeline of the Summer Analytics 2026 Capstone Project organised by the Consulting and Analytics Club at IIT Guwahati in collaboration with Celonis

# Part 1 - Travel Sustainability Portfolio

This part utilizes Celonis process intelligence to transform Scope 3 travel data into a quantified, actionable decarbonization roadmap. The primary goal is to address the bulk of Scope 3 emissions stemming from employee business travel and support the corporate commitment to achieving Net Zero by 2030. By analyzing the travel booking lifecycle, the project identifies high-emission process bottlenecks and replaces them with automated, sustainable alternatives.

## Key Insights & Executive Summary
The travel footprint was analyzed across 65,289 trips generated between December 2023 and June 2025.

* **Total Footprint:** Business travel generates approximately 114 million kg of CO2 annually, making it the largest identifiable Scope 3 category.
* **Financial Scope:** The total financial footprint amounts to 90 million in travel expenditure.
* **The Policy Leakage Problem:** Almost a quarter (24.4%) of emissions stem from just 18.4% of trips booked out-of-policy.
* **Root Cause of Inefficiency:** Late bookings drive high carbon costs; compliant trips average 22.1 days of lead time, whereas out-of-policy trips average just 6.2 days, forcing higher-carbon fares. 
* **Cabin Class Impact:** First-class flights emit roughly four times more CO2 per trip than economy, and business class emits three times more. Premium cabins account for 25% of flights but contribute to 45% of the total emissions.
* **Regional Hotspots:** Asia, Oceania, and South America are the biggest hotspots. Oceania and South America heavily drive the footprint due to long-haul hubs, accounting for over 70% of the total global footprint. Asia is the most carbon-intensive destination due to longer routes and limited rail options.

## Strategic Recommendations
Implementing the following strategies can recover approximately 26% of the annual footprint.

* **Right-Size Cabin Class:** Downsizing eligible Business and First Class flights to Economy can save roughly 4,000 kg of CO2 per trip. This policy shift is cost-neutral and yields an impact of -24,900 tCO2e/yr.
* **Electrify Rental Fleet:** Replacing petrol and diesel rental vehicles with EVs across top operating markets achieves an estimated 600 kg CO2 reduction per trip[cite: 1]. This results in an impact of -4,650 tCO2e/yr, requiring an additional $430K/yr via leasing.
* **Rail-First Policy:** Mandating train travel for high-frequency short routes (e.g., Berlin-Madrid) can save over 6,500 kg CO2 per trip and cut cash costs. This translates to a reduction of 300+ tCO2e/yr and saves $79K/yr.
* **Enforce 14-Day Window:** Implementing automated pre-approval blocks for flights booked under 14 days will directly mitigate policy leakage. This automation targets 43,500 tCO2e.

## Strategic Integration Roadmap
The decarbonization strategy is structured across three phases:

1. **Phase 1 (2026-27) - Quick Wins:** Enforce cabin-class policies, pilot EV rentals in the top 3 markets, and launch the rail-first mandate for the top 10 EU routes, generating a ~30M CO2e/yr reduction.
2. **Phase 2 (2027-28) - Integrate & Automate:** Embed live CO2e estimates directly into the booking flow and fully automate the 14-day pre-approval gate to definitively close policy leakage.
3. **Phase 3 (2029-30) - Scale & Offset:** Expand the rail-first policy to Asia and North America (infrastructure permitting) and utilize verified offsets for residual long-haul travel to achieve Net Zero.

## Data Architecture
The analysis was conducted within a Celonis Data Pool using only two out of three core datasets:

* `public_trip_data.csv`: The master data table containing trip details (locations, transport type, costs) and pre-calculated carbon emissions.
* `public_trip_event_log.csv`: The activity table containing timestamps for every step of a trip (Booking to Reimbursement), essential for Process Explorer.

## Dashboard
The system interface features a three-tab dashboard architecture to monitor progress:

* **Tab 1 - Overview:** Tracks high-level KPIs including Total Trips (65,289), Total CO2 Emissions (178M kg), Total Cost (140M), and the % of Trips Out-of-Policy (18.39%).
* **Tab 2 - Insights / Benchmarking:** Analyzes emissions by region, business unit, and travel purpose. Data reveals that Sales and Marketing combine to generate 51% of total emissions, driven strictly by trip volume. Customer Visits and Conferences account for 79% of the footprint.
* **Tab 3 - Process Analysis / Emissions Drilldown:** Utilizes process discovery tools to visualize the booking behavior and approval bottlenecks mapping the end-to-end trip lifecycle.

# Part 2 - High-Carbon Trip Predictor

This part involves building a ML model for classifying corporate travel as `HighCarbon`. Using a lean, highly effective feature set and a tuned `HistGradientBoostingClassifier`, this approach prioritizes signal over noise to achieve the highest predictive performance.

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
