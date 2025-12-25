import joblib
import requests
import json
from flask import Flask, request, render_template
import os

# Initialize Flask app
app = Flask(__name__)

# Load the Random Forest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Groq API configuration

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_ai_recommendations(air_temp, process_temp, rotational_speed, torque, prediction_result):
    """
    Get AI-powered maintenance recommendations based on machine parameters
    """
    try:
        # Create context-aware prompt
        maintenance_status = "requires maintenance" if prediction_result == 1 else "is in good condition"
        
        prompt = f"""
        As an industrial maintenance expert, analyze these machine parameters and provide 4 specific, actionable maintenance recommendations:
        
        Machine Status: {maintenance_status}
        - Air Temperature: {air_temp} K ({air_temp - 273.15:.1f}°C)
        - Process Temperature: {process_temp} K ({process_temp - 273.15:.1f}°C)
        - Rotational Speed: {rotational_speed} RPM
        - Torque: {torque} Nm
        
        Consider these factors in your analysis:
        - Temperature differentials and thermal stress
        - Rotational speed vs torque relationship
        - Wear patterns and lubrication needs
        - Energy efficiency optimization
        - Predictive maintenance strategies
        - Safety protocols and compliance
        
        Provide exactly 4 recommendations in this JSON format:
        {{
            "recommendations": [
                {{
                    "title": "Specific Action Title",
                    "description": "Detailed explanation of what to do and why",
                    "icon": "fas fa-relevant-icon",
                    "priority": "high/medium/low"
                }}
            ]
        }}
        
        Make recommendations specific to the actual parameter values, not generic advice. Include a mix of immediate actions, preventive measures, monitoring suggestions, and optimization opportunities.
        """
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert industrial maintenance engineer with 20+ years of experience in predictive maintenance systems."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1200
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                json_str = content[start_idx:end_idx]
                recommendations_data = json.loads(json_str)
                return recommendations_data['recommendations']
            except:
                # Fallback to default recommendations if JSON parsing fails
                return get_default_recommendations(air_temp, process_temp, rotational_speed, torque, prediction_result)
        else:
            print(f"Groq API error: {response.status_code}")
            return get_default_recommendations(air_temp, process_temp, rotational_speed, torque, prediction_result)
            
    except Exception as e:
        print(f"Error getting AI recommendations: {str(e)}")
        return get_default_recommendations(air_temp, process_temp, rotational_speed, torque, prediction_result)

def get_default_recommendations(air_temp, process_temp, rotational_speed, torque, prediction_result):
    """
    Fallback function to provide context-aware recommendations based on parameter analysis
    """
    recommendations = []
    
    # Temperature analysis
    temp_diff = process_temp - air_temp
    air_temp_c = air_temp - 273.15
    process_temp_c = process_temp - 273.15
    
    if temp_diff > 15:  # High temperature differential
        recommendations.append({
            "title": "Cooling System Optimization",
            "description": f"Temperature differential of {temp_diff:.1f}K detected. Check cooling efficiency, clean heat exchangers, and verify coolant flow rates to prevent thermal stress.",
            "icon": "fas fa-snowflake",
            "priority": "high"
        })
    elif process_temp_c > 50:  # High process temperature
        recommendations.append({
            "title": "Temperature Monitoring Enhancement",
            "description": f"Process temperature at {process_temp_c:.1f}°C requires attention. Implement continuous thermal monitoring and consider heat dissipation improvements.",
            "icon": "fas fa-thermometer-half",
            "priority": "medium"
        })
    else:
        recommendations.append({
            "title": "Thermal Stability Maintenance",
            "description": f"Current thermal conditions are stable. Monitor temperature trends and maintain cooling system efficiency to prevent future overheating.",
            "icon": "fas fa-temperature-low",
            "priority": "low"
        })
    
    # Speed and torque analysis
    power_estimate = (torque * rotational_speed * 2 * 3.14159) / 60000  # Approximate power in kW
    
    if rotational_speed > 2000:  # High speed operation
        recommendations.append({
            "title": "High-Speed Bearing Maintenance",
            "description": f"Operating at {rotational_speed} RPM requires premium lubrication. Schedule bearing inspection and use high-speed compatible lubricants.",
            "icon": "fas fa-tachometer-alt",
            "priority": "high"
        })
    elif rotational_speed < 1000:  # Low speed, check for efficiency
        recommendations.append({
            "title": "Low-Speed Operation Analysis",
            "description": f"Low speed operation at {rotational_speed} RPM may indicate efficiency issues. Check for mechanical resistance and alignment problems.",
            "icon": "fas fa-search",
            "priority": "medium"
        })
    else:
        recommendations.append({
            "title": "Optimal Speed Range Monitoring",
            "description": f"Current speed of {rotational_speed} RPM is within normal range. Continue monitoring for speed variations and maintain consistent operation.",
            "icon": "fas fa-gauge",
            "priority": "low"
        })
    
    if torque > 50:  # High torque
        recommendations.append({
            "title": "High-Torque Component Inspection",
            "description": f"High torque load of {torque} Nm detected. Inspect coupling alignment, check for mechanical stress, and verify fastener torque specifications.",
            "icon": "fas fa-wrench",
            "priority": "high"
        })
    elif torque < 20:  # Low torque might indicate slipping
        recommendations.append({
            "title": "Drive System Verification",
            "description": f"Low torque reading of {torque} Nm may indicate slipping or reduced load transfer. Check belt tension and coupling integrity.",
            "icon": "fas fa-tools",
            "priority": "medium"
        })
    else:
        recommendations.append({
            "title": "Torque Load Optimization",
            "description": f"Current torque of {torque} Nm is within acceptable range. Monitor for load variations and optimize power transfer efficiency.",
            "icon": "fas fa-balance-scale",
            "priority": "low"
        })
    
    # Maintenance prediction specific recommendations
    if prediction_result == 1:  # Maintenance required
        recommendations.append({
            "title": "Immediate Maintenance Protocol",
            "description": "AI model indicates maintenance required. Schedule immediate inspection focusing on wear components, lubrication levels, and alignment checks.",
            "icon": "fas fa-exclamation-triangle",
            "priority": "high"
        })
    else:  # No maintenance required
        recommendations.append({
            "title": "Preventive Maintenance Scheduling",
            "description": "Machine is operating normally. Maintain current maintenance schedule and monitor parameter trends for early detection of changes.",
            "icon": "fas fa-calendar-check",
            "priority": "low"
        })
    
    # Always add these comprehensive recommendations to reach 4
    additional_recommendations = [
        {
            "title": "Vibration Analysis Implementation",
            "description": "Conduct comprehensive vibration analysis to detect early signs of mechanical wear, misalignment, or bearing degradation before they cause failures.",
            "icon": "fas fa-wave-square",
            "priority": "medium"
        },
        {
            "title": "Lubrication Schedule Optimization",
            "description": "Review and optimize lubrication intervals based on current operating conditions, load factors, and manufacturer specifications for maximum efficiency.",
            "icon": "fas fa-oil-can",
            "priority": "medium"
        },
        {
            "title": "Performance Trending & Analytics",
            "description": "Establish comprehensive baseline performance metrics and implement advanced trend analysis for predictive maintenance optimization.",
            "icon": "fas fa-chart-line",
            "priority": "low"
        },
        {
            "title": "Energy Efficiency Audit",
            "description": f"With estimated power output of {power_estimate:.2f} kW, conduct energy efficiency audit to identify optimization opportunities and reduce operational costs.",
            "icon": "fas fa-bolt",
            "priority": "medium"
        },
        {
            "title": "Safety Protocol Review",
            "description": "Review current safety protocols and emergency procedures to ensure compliance with latest industrial safety standards and regulations.",
            "icon": "fas fa-shield-alt",
            "priority": "medium"
        },
        {
            "title": "Sensor Calibration Check",
            "description": "Verify accuracy of temperature, speed, and torque sensors through calibration checks to ensure reliable data for predictive maintenance.",
            "icon": "fas fa-adjust",
            "priority": "low"
        }
    ]
    
    # Ensure we have exactly 4 recommendations
    while len(recommendations) < 4:
        # Add from additional recommendations pool
        for rec in additional_recommendations:
            if rec not in recommendations and len(recommendations) < 4:
                recommendations.append(rec)
                break
    
    # If we have more than 4, prioritize by priority level
    if len(recommendations) > 4:
        high_priority = [r for r in recommendations if r["priority"] == "high"]
        medium_priority = [r for r in recommendations if r["priority"] == "medium"]
        low_priority = [r for r in recommendations if r["priority"] == "low"]
        
        # Select 4 recommendations with priority distribution
        final_recommendations = []
        
        # Add high priority first (max 2)
        final_recommendations.extend(high_priority[:2])
        
        # Add medium priority to fill remaining slots
        remaining_slots = 4 - len(final_recommendations)
        final_recommendations.extend(medium_priority[:remaining_slots])
        
        # Fill any remaining slots with low priority
        remaining_slots = 4 - len(final_recommendations)
        if remaining_slots > 0:
            final_recommendations.extend(low_priority[:remaining_slots])
        
        recommendations = final_recommendations
    
    return recommendations[:4]

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        feature_1 = float(request.form['feature_1'])  # Air Temperature
        feature_2 = float(request.form['feature_2'])  # Process Temperature  
        feature_3 = float(request.form['feature_3'])  # Rotational Speed
        feature_4 = float(request.form['feature_4'])  # Torque

        # Print the features for debugging
        print(f"Features: {feature_1}, {feature_2}, {feature_3}, {feature_4}")

        # Make prediction using the Random Forest model
        result = random_forest_model.predict([[feature_1, feature_2, feature_3, feature_4]])[0]
        
        # Print the result for debugging
        print(f"Prediction Result: {result}")

        # Convert the prediction result to a readable format
        prediction_text = "Maintenance Required" if result == 1 else "No Maintenance Required"
        
        # Get AI-powered recommendations (now returns 4)
        recommendations = get_ai_recommendations(feature_1, feature_2, feature_3, feature_4, result)
        
        # Prepare additional context for the template
        context = {
            'air_temp_c': round(feature_1 - 273.15, 1),
            'process_temp_c': round(feature_2 - 273.15, 1),
            'temp_diff': round(feature_2 - feature_1, 1),
            'power_estimate': round((feature_4 * feature_3 * 2 * 3.14159) / 60000, 2)
        }

        # Render the result page with the prediction and recommendations
        return render_template('result.html', 
                             prediction=prediction_text,
                             recommendations=recommendations,
                             context=context,
                             parameters={
                                 'air_temp': feature_1,
                                 'process_temp': feature_2,
                                 'rotational_speed': feature_3,
                                 'torque': feature_4
                             })

    except Exception as e:
        # If an error occurs, return the error message
        print(f"Error in predict route: {str(e)}")
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)