# Vietnam Holiday Feature Audit

Scope: holiday/event dates used for forecasting features from 2012 through the
forecast horizon ending 2024-07-01.  This audit treats holiday dates as public
calendar knowledge and does not use target values from `sample_submission.csv`.

## Sources

- Vietnam Briefing says Vietnam public holidays are administered by MoLISA and
  gives the 2024 official public holiday schedule, including Tet, Hung Kings,
  April 30, May 1, and National Day.
- Office Holidays provides year-specific Vietnam public holiday tables.  Its
  2024 table lists Jan 1, Tet holiday dates, Hung Kings on Apr 18, Apr 30,
  May 1, and Sep 2-3.
- Calendarific was used as a cross-check for older years where Office Holidays
  pages are incomplete or inconsistent.
- VOV/VietnamPlus reports confirm that the 2018 Hung Kings main day was
  Apr 25, 2018, the 10th day of the third lunar month.

Primary URLs:

- https://www.vietnam-briefing.com/news/vietnams-public-holidays.html/
- https://www.officeholidays.com/countries/vietnam/2024
- https://calendarific.com/holidays/2012/VN
- https://calendarific.com/holidays/2013/VN
- https://calendarific.com/holidays/2014/VN
- https://english.vov.vn/en/society/crowds-flood-hung-kings-temple-for-national-festival-373241.vov
- https://en.vietnamplus.vn/ceremonies-commemorate-hung-kings-nationwide-post130087.vnp

## Canonical Feature Table

The machine-readable table is saved at:

```text
docs/vietnam_holiday_calendar_2012_2024.csv
```

Columns:

- `tet_nguyen_dan`: lunar Jan 1, core Tet date.
- `tet_eve`: one day before lunar Jan 1.
- `hung_kings_10_3_lunar`: 10th day of the 3rd lunar month.
- `mid_autumn_15_8_lunar`: 15th day of the 8th lunar month.
- `black_friday`: last Friday of November.
- `cyber_monday`: Monday after Black Friday.
- Fixed-date events: New Year, Apr 30, May 1, Sep 2, Valentine, Mar 8, Oct 20,
  Teachers' Day, Christmas.

Observed/substitute public days can vary by government decision and weekend
compensation.  The model features therefore use both exact event-day flags and
windows around major events rather than relying only on observed days off.

## Code Audit

File audited: `src/calendar_vn.py`.

Findings:

1. Tet dates for 2012-2024 were correct.
2. Mid-Autumn dates for 2012-2024 were correct.
3. Hung Kings dates were missing from the feature file.
4. `is_cyber_monday_window` was effectively wrong because it checked negative
   values on `days_to_black_friday`, but that array is clipped to non-negative
   distances.
5. Exact-day flags were missing for Tet, Mid-Autumn, and Black Friday.
6. Office Holidays has an incorrect 2018 Hung Kings row in one scrape path
   (`2018-03-27`).  Cross-checks show the correct date is `2018-04-25`.

Fixes applied:

- Added `HUNG_KINGS` dates for 2012-2024.
- Added `days_to_hung_kings`, `days_since_hung_kings`,
  `is_hung_kings_day`, `is_hung_kings_eve_7d`,
  `is_hung_kings_after_3d`, and `is_hung_kings_window`.
- Added exact flags: `is_tet_nguyen_dan`, `is_mid_autumn_day`,
  `is_black_friday`.
- Added `days_since_black_friday`.
- Fixed `is_cyber_monday_window` to use `days_since_black_friday` from 1 to 3
  days after Black Friday.

## Modeling Notes

For this dataset, the likely useful holiday signals are:

- Tet: demand suppression/closure before and after lunar New Year.
- Hung Kings: public-holiday effect around March/April; possible bridge days
  with Apr 30/May 1 when dates are close.
- Apr 30 and May 1: national/public-holiday shopping and logistics effects.
- Sep 2: National Day and possible extended holiday since 2021.
- Mid-Autumn: children/family gifting effect; likely category dependent.
- 11/11, Black Friday/Cyber Monday, 12/12: e-commerce promo intensity.
- Mar 8, Oct 20, Nov 20, Christmas, Valentine: gift/fashion category events.

