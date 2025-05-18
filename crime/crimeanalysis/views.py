

# Load model, encoder and save your feature columns list during training and load here
import pandas as pd
import os
import pickle
import numpy as np
from django.shortcuts import render
from django.conf import settings

model = pickle.load(open(r'C:\Users\gunda\OneDrive\Desktop\crime_prediction\model\rfclassifier.pkl', 'rb'))

encoder = pickle.load(open(r'C:\Users\gunda\OneDrive\Desktop\crime_prediction\model\encoder.pkl', 'rb'))

# Define all feature columns after one-hot encoding, exactly in order your model expects
feature_columns = ['Latitude', 'Longitude', 'Hour', 'Crimes_Last_30_Days',
                   'Distance_to_Police_Station_km', 'Year',
                   'City_Bengaluru', 'City_Chennai', 'City_Delhi', 'City_Hyderabad',
                   'City_Jaipur', 'City_Kolkata', 'City_Mumbai', 'City_Pune',
                   'Day_of_Week_Friday', 'Day_of_Week_Monday', 'Day_of_Week_Saturday',
                   'Day_of_Week_Sunday', 'Day_of_Week_Thursday', 'Day_of_Week_Tuesday',
                   'Day_of_Week_Wednesday',
                   'Population_Density_High', 'Population_Density_Low',
                   'Population_Density_Medium', 'Population_Density_Very High',
                   'Area_Type_Commercial', 'Area_Type_Mixed', 'Area_Type_Residential',
                   'Nearby_Facility_ATM', 'Nearby_Facility_Bar', 'Nearby_Facility_Mall',
                   'Nearby_Facility_Park', 'Nearby_Facility_School', 'Nearby_Facility_Unknown']




def prediction(request):
    if request.method == 'POST':
        # Extract raw inputs from POST
        city = request.POST.get('City')
        latitude = float(request.POST.get('Latitude', 0))
        longitude = float(request.POST.get('Longitude', 0))
        day_of_week = request.POST.get('Day_of_Week')
        hour = int(request.POST.get('Hour', 0))
        crimes_last_30_days = int(request.POST.get('Crimes_Last_30_Days', 0))
        population_density = request.POST.get('Population_Density')
        area_type = request.POST.get('Area_Type')
        distance_to_police_station_km = float(request.POST.get('Distance_to_Police_Station_km', 0))
        nearby_facility = request.POST.get('Nearby_Facility')
        year = int(request.POST.get('Year', 2025))

        # Build input vector with zeros for all features first
        input_vector = {col: 0 for col in feature_columns}

        # Fill numeric features
        input_vector['Latitude'] = latitude
        input_vector['Longitude'] = longitude
        input_vector['Hour'] = hour
        input_vector['Crimes_Last_30_Days'] = crimes_last_30_days
        input_vector['Distance_to_Police_Station_km'] = distance_to_police_station_km
        input_vector['Year'] = year

        # Helper to set one-hot encoded categorical feature, if exists
        def set_one_hot(feature_prefix, feature_value):
            key = f"{feature_prefix}_{feature_value}"
            if key in input_vector:
                input_vector[key] = 1
            # else: leave all zeros (handle unknown category)

        # Set categorical features
        set_one_hot('City', city)
        set_one_hot('Day_of_Week', day_of_week)
        set_one_hot('Population_Density', population_density)
        set_one_hot('Area_Type', area_type)
        set_one_hot('Nearby_Facility', nearby_facility)

        # Prepare input array in correct order
        input_dict = {col: input_vector.get(col, 0) for col in feature_columns}  # get from input_vector or default 0

# Create DataFrame with exactly the model columns, in order
        input_df = pd.DataFrame([input_dict], columns=feature_columns)

        prediction = model.predict(input_df)[0]
        risk_mapping = {0: "Low", 1: "Medium", 2: "High"}
        decoded_prediction = risk_mapping.get(prediction, "Unknown")

        # Pass prediction to template
        return render(request, 'prediction.html', {'prediction':decoded_prediction})

    else:
        # GET request - just render form
        return render(request, 'prediction.html')

