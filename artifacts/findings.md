# H&M Fashion Intelligence — findings

Historical H&M transactions, 2018-09-20 00:00:00 to 2020-09-22 00:00:00. Prices are anonymised, so value is relative, not a currency.

## Headline numbers
- Purchase events: **28,813,419**
- Customers: **1,362,281**
- Repeat customers (more than one visit): **67.1%**
- Average visits per customer: **6.67**
- Median customer age: **32**
- Online share of purchase events: **31.7%**

## Top departments by purchase events
- Swimwear: 2,128,005
- Trouser: 1,549,348
- Blouse: 1,519,154
- Knitwear: 1,505,655
- Jersey: 1,390,422
- Jersey Basic: 1,301,964
- Expressive Lingerie: 1,084,735
- Jersey fancy: 1,078,516
- Basic 1: 1,027,967
- Dress: 1,008,274

## Data-quality observations
- transactions: exact duplicate rows removed: 2,974,905 (info)
- customers: missing age: 15,861 (warning)
- customers: never purchased: 9,699 (warning)
- transactions: articles never sold: 995 (warning)

## Honest limitations
- H&M anonymised and scaled the prices, so value here is relative, not a currency amount.
- The dataset has no order id, so a (customer, date) visit is a documented proxy for a basket, not a true order.
- Exact duplicate rows were removed; because there is no quantity column, some may have represented multiple units.
- There is no delivery, review, payment or seller information, so those analyses are not possible.
- This is historical data (2018-2020) and not a live feed.
