# Team-Grease-IBMZDatathon2026
SSF2026

## The problem
Medical emergencies sometimes require a response time faster than is possible with traditional ambulance services; particularly in New York City, where heavy traffic stopping large ambulances is common

## Why this problem matters

**Response times are getting worse.**
- NYC response times for life-threatening medical emergencies rose from 9:34 in 2021 to 13:09 in the latest fiscal year reported ([EMS1](https://www.ems1.com/response-times/nyc-ambulance-response-times-hit-record-high-amid-ems-staffing-shortage)).
- FDNY's EMS chief told the City Council the department loses about 10 in-service ambulances a day because of staffing shortages (same source).

**The burden is uneven across boroughs.**
- The NYC Independent Budget Office found the share of advanced life support calls answered within 10 minutes fell in every borough between 2014 and 2024. In 2024 it was about 82% in Staten Island and about 75% in the Bronx. For basic life support, the Bronx was about 48% against about 68% in Staten Island ([IBO analysis summary](https://isoclivecivic.substack.com/p/exploring-an-independent-perspective)).

**Minutes decide survival in cardiac arrest.**
- The American Heart Association rule of thumb is that survival chances fall by 7-10% for each minute without CPR or defibrillation ([CARES registry report](https://mycares.net/sitepages/uploads/2024/2023_flipbook/inc/html/34.html)). Newer studies refine this, showing much better survival with defibrillation within about four minutes ([FireRescue1](https://www.firerescue1.com/fire-products/aeds/articles/ny-countys-cardiac-emergency-response-time-subpar-ISf1uNeBLRnlC7jS/)).

**Recognising the emergency on the phone is part of the problem.**
- Dispatchers fail to identify roughly 25% of out-of-hospital cardiac arrests ([study abstract](https://swov.nl/en/publicatie/machine-learning-supportive-tool-recognize-cardiac-arrest-emergency-calls)).
- In a Copenhagen study of 108,607 calls, a machine learning model recognised cardiac arrest more often and faster than dispatchers, but with more false alarms (same source).

**Why we use decision support, not automation.**
- A randomised trial across 169,049 calls found that alerting dispatchers did not significantly improve their recognition of cardiac arrest, even though the model outperformed humans on its own ([trial summary](https://busqueda.bvsalud.org/portal/resource/es/mdl-33404620)). We keep a human dispatcher in the loop and test our system by simulation.

**Data limitation.**
- NYC's public EMS dispatch data aggregates incident locations to protect patient privacy, so our analysis works at area level, not street level ([data.gov](https://catalog.data.gov/dataset/ems-incident-dispatch-data)).



