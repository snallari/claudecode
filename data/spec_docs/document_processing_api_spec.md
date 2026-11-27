# Document Processing System – API Specification

**Version:** `1.0.0-draft`  
**Architecture Style:** Event-driven, asynchronous processing via Kafka  
**Primary Goal:** Ingest documents, process asynchronously, persist validated output, and expose retrieval APIs driven by database metadata/configuration.

## 1) Scope

This specification defines:
- External APIs for document ingestion and retrieval.
- Event contracts for Kafka producer/consumer flows.
- Database-driven dynamic API metadata model.
- Validation, idempotency, retry/DLQ, security, observability, and error semantics.

This specification does not define UI behavior.

## 2) End-to-End Flow

1. Client uploads document or references external source.
2. API stores file in object storage, writes metadata in DB, marks status `RECEIVED` -> `PENDING`.
3. Producer publishes event to `document.processing`.
4. Consumer receives event, loads document from storage reference.
5. Consumer parses, transforms, validates document by type-specific rules.
6. Consumer persists extracted data + processing history in a transaction.
7. Status transitions to `PROCESSED` or `FAILED`/`RETRY_PENDING`.
8. Runtime API engine reads API definitions from DB and serves configured endpoints.

## 3) Processing Status Lifecycle

Allowed values:
- `RECEIVED`
- `PENDING`
- `PROCESSING`
- `PROCESSED`
- `FAILED`
- `RETRY_PENDING`

Recommended transitions:
- `RECEIVED -> PENDING -> PROCESSING -> PROCESSED`
- `PROCESSING -> RETRY_PENDING -> PROCESSING`
- `PROCESSING -> FAILED` (after max retries)

## 4) API Surface (External)

### 4.1 Upload Document

`POST /api/v1/documents`

Purpose:
- Accept document upload.
- Create metadata + object storage reference.
- Publish processing event asynchronously.

Request:
- `multipart/form-data`
	- `file` (required)
	- `sourceSystem` (optional, string)
	- `documentType` (optional; inferred if omitted)

Response `202 Accepted`:

```json
{
	"documentId": "doc_01JAZQ8H2Q1S7YF8V2P1XK8M3N",
	"status": "PENDING",
	"uploadedAt": "2026-11-26T23:00:00Z",
	"fileLocation": "s3://docs-bucket/2026/11/26/doc_...pdf",
	"correlationId": "corr_8f3c..."
}
```

Errors:
- `400` invalid format/size
- `401/403` auth/authorization
- `409` duplicate idempotency key conflict
- `500` persistence/publish failure

### 4.2 List Documents

`GET /api/v1/documents`

Query params:
- `page` (default `1`), `pageSize` (default `20`, max `200`)
- `status`, `documentType`, `sourceSystem`
- `fromUploadedAt`, `toUploadedAt`
- `sortBy` (`uploadedAt`, `documentName`, `status`), `sortOrder` (`asc`, `desc`)
- `search` (name/id contains)

Response `200`:

```json
{
	"items": [
		{
			"documentId": "doc_...",
			"documentName": "invoice_2241.pdf",
			"documentType": "PDF",
			"status": "PROCESSED",
			"uploadedAt": "2026-11-26T23:00:00Z",
			"sourceSystem": "PORTAL"
		}
	],
	"page": 1,
	"pageSize": 20,
	"total": 125
}
```

### 4.3 Get Document Metadata

`GET /api/v1/documents/{documentId}`

Response `200`: metadata + latest processing summary.

### 4.4 Get Document Status

`GET /api/v1/documents/{documentId}/status`

Response `200`:

```json
{
	"documentId": "doc_...",
	"status": "PROCESSING",
	"lastUpdatedAt": "2026-11-26T23:01:10Z",
	"retryCount": 1,
	"correlationId": "corr_..."
}
```

### 4.5 Get Processed Content

`GET /api/v1/documents/{documentId}/content`

Response `200`:

```json
{
	"documentId": "doc_...",
	"schemaVersion": "1.2",
	"content": {
		"fields": {
			"invoiceNumber": "INV-2241",
			"amount": 320.45
		}
	}
}
```

### 4.6 Reprocess Failed Document

`POST /api/v1/documents/{documentId}/reprocess`

Behavior:
- Allowed when status in `FAILED` or `RETRY_PENDING`.
- Publishes to `document.retry` or `document.processing`.

Response `202 Accepted`.

## 5) Kafka Event Contracts

## 5.1 Topic Names

- `document.received`
- `document.processing`
- `document.processed`
- `document.failed`
- `document.retry`
- `document.dlq`

### 5.2 Processing Event Payload

```json
{
	"eventId": "evt_01JAZ...",
	"eventType": "DOCUMENT_PROCESS_REQUESTED",
	"eventTimestamp": "2026-11-26T23:00:03Z",
	"correlationId": "corr_8f3c...",
	"processingAction": "PROCESS",
	"document": {
		"documentId": "doc_01JAZ...",
		"documentType": "PDF",
		"documentLocation": "s3://docs-bucket/...pdf",
		"sourceSystem": "PORTAL"
	},
	"trace": {
		"producerService": "document-ingestion-service",
		"version": "1.0.0"
	}
}
```

Headers:
- `correlationId`
- `idempotencyKey` (optional)
- `schemaVersion`

Delivery requirements:
- Producer acks `all`, retries enabled, idempotent producer on.
- At-least-once consumption.

## 6) Idempotency and Duplicate Handling

- API upload supports `Idempotency-Key` header.
- DB uniqueness constraints:
	- `DOCUMENT_METADATA.document_id` unique.
	- `DOCUMENT_PROCESSING_HISTORY(event_id)` unique.
- Consumer upsert logic by `document_id` + version key.
- Replayed Kafka events must not create duplicate records.

## 7) Data Model (Logical)

## 7.1 `DOCUMENT_METADATA`
- `document_id` (PK)
- `document_name`
- `document_type`
- `file_location`
- `uploaded_timestamp`
- `processing_status`
- `source_system`
- `correlation_id`
- `created_at`, `updated_at`

## 7.2 `DOCUMENT_DATA`
- `document_data_id` (PK)
- `document_id` (FK)
- `schema_version`
- `extracted_payload` (JSON/JSONB)
- `validation_result`
- `created_at`, `updated_at`

## 7.3 `DOCUMENT_PROCESSING_HISTORY`
- `history_id` (PK)
- `document_id` (FK)
- `event_id`
- `status_from`, `status_to`
- `message`
- `processed_at`

## 7.4 `DOCUMENT_PROCESSING_ERRORS`
- `error_id` (PK)
- `document_id` (FK)
- `event_id`
- `error_code`
- `error_message`
- `stack_trace` (optional)
- `retry_count`
- `recorded_at`

## 7.5 Dynamic API Tables

- `API_DEFINITION(API_ID, API_NAME, API_PATH, HTTP_METHOD, API_VERSION, STATUS)`
- `API_REQUEST_MAPPING(MAPPING_ID, API_ID, PARAMETER_NAME, PARAMETER_TYPE, IS_REQUIRED)`
- `API_RESPONSE_MAPPING(MAPPING_ID, API_ID, RESPONSE_FIELD, DATABASE_COLUMN, DATA_TYPE)`
- `API_DATABASE_MAPPING(MAPPING_ID, API_ID, TABLE_NAME, STORED_PROCEDURE, QUERY_CONFIGURATION)`
- `API_SCHEMA(SCHEMA_ID, API_ID, REQUEST_SCHEMA, RESPONSE_SCHEMA)`

## 8) Dynamic API Runtime Behavior

At startup (and periodic refresh), runtime loads active definitions:
- Only `API_DEFINITION.STATUS='ACTIVE'`.
- Resolves path/method/version conflicts.
- Validates request schemas before query execution.
- Maps query result columns to response fields using `API_RESPONSE_MAPPING`.

Execution pipeline per request:
1. Resolve endpoint metadata from in-memory registry.
2. Validate request (path/query/body) against `API_SCHEMA.REQUEST_SCHEMA`.
3. Build SQL/stored-procedure call using safe parameterization.
4. Execute against DB.
5. Transform response via mapping table.
6. Validate against `API_SCHEMA.RESPONSE_SCHEMA`.

## 9) Error Model

Standard error envelope:

```json
{
	"error": {
		"code": "DOCUMENT_NOT_FOUND",
		"message": "Document does not exist",
		"details": null,
		"correlationId": "corr_8f3c...",
		"timestamp": "2026-11-26T23:05:00Z"
	}
}
```

Common codes:
- `INVALID_REQUEST`
- `UNSUPPORTED_DOCUMENT_TYPE`
- `DOCUMENT_NOT_FOUND`
- `PROCESSING_NOT_COMPLETE`
- `RETRY_LIMIT_EXCEEDED`
- `INTERNAL_ERROR`

## 10) Retry, DLQ, and Recovery

- Consumer retry policy: exponential backoff (`maxRetries` configurable).
- On retryable failures -> publish to `document.retry` and set status `RETRY_PENDING`.
- On permanent failure -> publish to `document.dlq` and set status `FAILED`.
- Persist all failures in `DOCUMENT_PROCESSING_ERRORS`.

## 11) Security

- OAuth2/JWT for all APIs.
- RBAC:
	- `document.write` for upload/reprocess
	- `document.read` for retrieval
	- `api.config.read/write` for dynamic API administration
- TLS for API/Kafka/DB in transit.
- Encryption at rest for object storage and database.
- Secrets stored in managed secret store.

## 12) Observability

Required structured log fields:
- `documentId`, `correlationId`, `eventId`, `status`, `service`, `durationMs`

Metrics:
- Ingestion requests/sec, upload failures
- Consumer lag by partition/group
- Processing throughput and p95 latency
- Retry count, DLQ rate
- DB write failures and query latency

Tracing:
- Propagate `correlationId` from upload -> Kafka -> consumer -> API responses.

## 13) API Versioning and Compatibility

- URI versioning: `/api/v1/...`
- Dynamic API table field `API_VERSION` controls route registration.
- Breaking changes require new version.
- Backward-compatible response extension allowed.

## 14) Acceptance Mapping

This spec satisfies requested E2E outcomes:
- Upload + unique document ID: Sections 4.1, 7.1
- Kafka publish/consume: Sections 5, 2
- Parse/validate/persist/status updates: Sections 2, 3, 7
- Duplicate-safe processing: Section 6
- Retry + DLQ: Section 10
- DB-driven dynamic API: Sections 7.5, 8
- Pagination/filtering/validation/error handling: Sections 4.2, 8, 9
- Traceability by document/correlation ID: Sections 5, 12

## 15) Implementation Notes

- Keep ingestion, processor, and dynamic API runtime as separate deployable services.
- Use schema registry when Avro is selected.
- Use transactional boundaries around metadata + data + history writes.
- Add DB indexes on `document_id`, `processing_status`, `uploaded_timestamp`, `correlation_id`.