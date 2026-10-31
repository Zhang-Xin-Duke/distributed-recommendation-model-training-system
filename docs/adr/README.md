# Architecture decisions

These decisions define the proposed implementation boundary.

| Record | Decision |
|---|---|
| [001](001-service-boundary.md) | Extract the embedding backend while retaining the training framework |
| [002](002-update-and-placement.md) | Preserve original sparse updates and move complete row state |
| [003](003-cost-based-placement.md) | Plan access and cache placement using end-to-end cost |
| [004](004-transport-and-deployment.md) | Use a C++ service and an in-process GPU agent |
