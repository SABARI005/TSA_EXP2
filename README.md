### DEVELOPED BY: SABARI S
### REGISTER NO: 212222240085

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
 Import necessary libraries (NumPy, Matplotlib)
 Load the dataset
 Calculate the linear trend values using least square method
 Calculate the polynomial trend values using least square method
 End the program
 
### PROGRAM:
A - LINEAR TREND ESTIMATION
```python
# LINEAR TREND ESTIMATION
# LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/data_date.csv')
data['Date'] = pd.to_datetime(data['Date'])
price = data.groupby('Date')['AQI Value'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(price))
y = price['AQI Value']
linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(price['Date'], price['AQI Value'], label='Original Data', marker='o')
plt.plot(price['Date'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('AQI Value')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

B- POLYNOMIAL TREND ESTIMATION
```python
# POLYNOMIAL TREND ESTIMATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/data_date.csv') #Removed nrows=50 to increase the number of data points
data['Date'] = pd.to_datetime(data['Date'])
price = data.groupby('Date')['AQI Value'].mean().reset_index()

# Polynomial trend estimation (degree 2)
x = np.arange(len(price))
y = price['AQI Value']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(price['Date'], price['AQI Value'], label='Original Data', marker='o')
plt.plot(price['Date'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('AQI Value')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

### OUTPUT
A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/cd0b6788-879e-4636-87d5-ab035de325b9)



B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/448d36cc-6038-424a-b43b-2f0c87d0e6cf)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
