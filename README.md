# Radiotherapy scheduling problem research

In this repo you can find code for various tests about the radiotherapy scheduling problem.  
Constraints taken into account :
- [x] Several machines type that treat different tumor locations with different times.
- [x] Machines preferred, allowed or forbidden.
- [x] Patients with different treatment periods (every day, every two days or twice a day).
- [x] Machines unavailability (already scheduled sessions & maintenance)
- [x] Enforce all the patient's fractions (*can be a soft constraint to deal with unfeasabilities*).
- [x] Ready date and due date, variable according to each patient's emergency.

Objectives :
1. Minimize days of delays (modeled as sum of tardiness).
2. Minimize treatment range, as periods enforcement is a soft constraint.
3. Maximize sessions on machines preferred.
4. Maximize treatment stability (i.e. sessions planned at the same hours each day).
5. Balance machines charge.

Improvements and ideas :
- [ ] Time windows preferences and add it as an objective.
- [ ] Patients rescheduling : allows patients to be rescheduled while they have not been called -> difficulties to manage the timetable.
- [ ] Allow machine changes for identical machines only.
- [ ] Dummy patients to take into account potential emergencies.
- [ ] Change sum of tardiness for a max.

## Deterministic models

## Stochastic models
