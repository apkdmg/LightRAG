# Token Insights

An admin-only tab in the WebUI that shows how many LLM tokens are used, what they
cost, and why, organised by purpose: **Extraction**, **Vision**, **Keyword** and
**Query**. It is read-only apart from its own settings. Models and keys stay on
the Providers page.

## Turning it on

Insights stores usage in Postgres, so it needs `LIGHTRAG_KV_STORAGE=PGKVStorage`.
With other storage the server runs as before and the tab says it is off. Tables
named `LIGHTRAG_USAGE_*` are created on first start.

| Setting | Default | Meaning |
| --- | --- | --- |
| `INSIGHTS_TIMEZONE` | `Asia/Kuala_Lumpur` | The timezone for days and hours |
| `INSIGHTS_PUBLIC_URL` | unset | Server address for the link in alert emails |

Alert and digest emails use the same `ALERT_EMAIL_TO` and `SMTP_*` settings as
provider-rejection alerts.

Open **Settings** on the tab to enter:

- **Prices** in USD per million tokens, separately for prompt and answer. BYO models
  need prices too. Tokens on a model with no price are counted and reported as such.
- **Budget**, which is shared by all workspaces and includes BYO usage.
- **Provider limit**, used to show room under the rate limit.
- **MYR rate**, used in the procurement brief.
- **Invoices**, used to compare the system key's usage with the provider's bill.
- **Estimate past usage**, which estimates earlier usage from the LLM cache once.
  Those days are marked as estimated and stay out of anomaly checks.

## What is recorded

One row per provider call or LLM cache hit: workspace, purpose, operation, source,
client, model, who pays (system key or BYO), token counts, the cost if the provider
reports one, latency and status. **Prompt and answer text is never stored.**

- Bindings `openai`, `azure_openai` and `gemini` report real token counts. Others
  are estimated with the tokenizer and counted as estimated. Vision with no provider
  figure is shown as not measured, because a word count cannot measure images.
- A failed usage write never affects the LLM call. Rows are buffered and retried;
  once the buffer is full the page shows how many were lost.
- Calls are kept for 90 days, hourly totals for 2 years and daily totals indefinitely.
- Deleting a workspace removes its name from usage. Its numbers stay under a short
  code, so totals and the bill check stay correct.
- Document names are partly hidden on the page and in exports.

## What needs time

| Feature | Starts after |
| --- | --- |
| Totals and charts | The first call |
| Anomaly checks | 14 days |
| Forecast | 28 days, with wider ranges until 8 weeks |
| Before and after check for a model change | 7 days after the change |

## Reading an anomaly

"Usual" is the median for the same weekday over the previous 8 weeks. An anomaly
says why a purpose moved: more work, more tokens per unit of work, or the cache
reusing less. Consecutive unusual days of the same kind are one item, so marking it
as expected still holds while it continues.

## API

All routes are under `/admin/insights` and require the admin role.
Send `X-LightRAG-Client: <name>` on queries to have them grouped under that client.
