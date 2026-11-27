# Document Processing System – Functional & Technical Requirements

## 1. Objective

Build an event-driven document processing system that receives documents, processes them asynchronously using Apache Kafka, persists the processed document data into a database, and dynamically exposes APIs based on database definitions and configurations.

The solution should support scalability, reliability, extensibility, and efficient retrieval of processed document information.

## 2. High-Level Architecture

**Processing flow:**

1. Document is uploaded or received by the application.
2. Document metadata and storage reference are created.
3. A document processing event is published to Kafka.
4. Kafka consumers subscribe to the document processing topic.
5. The consumer retrieves and processes the document.
6. Extracted and transformed data is validated.
7. Processed information is persisted into the database.
8. API definitions and configurations are maintained in the database.
9. API runtime reads database metadata and exposes the corresponding APIs.
10. Consumers can retrieve document information through the generated APIs.

## 3. Functional Requirements

### FR-01: Document Ingestion

- The system shall accept documents through an upload API or an existing document source.
- Support PDF, DOCX, JSON, and other agreed document formats.
- Generate a unique document ID for every document.
- Capture document metadata:
  - Document ID
  - Document name
  - Document type
  - File location
  - Uploaded timestamp
  - Processing status
  - Source system
- Store the original document in a designated file/object storage location.
- Avoid storing large binary documents directly in Kafka messages.

### FR-02: Kafka Event Publishing

- Publish a document processing event to a designated Kafka topic.
- Include the following information in the event:
  - Event ID
  - Document ID
  - Document location/reference
  - Document type
  - Event timestamp
  - Processing action
  - Correlation ID
- Use JSON or Avro as the event serialization format.
- Configure Kafka producer acknowledgment and retry mechanisms.
- Ensure events are not lost during temporary downstream failures.

### FR-03: Kafka Consumer and Document Processing

- Implement a Kafka consumer service to subscribe to document processing events.
- Retrieve the original document using the document reference.
- Parse and extract the required document content.
- Transform extracted content into a standardized data structure.
- Perform schema and business validation.
- Support configurable processing rules based on document type.
- Update the document processing status.

**Processing statuses:**
- RECEIVED
- PENDING
- PROCESSING
- PROCESSED
- FAILED
- RETRY_PENDING

### FR-04: Database Persistence

- Persist processed document information into the database.
- Maintain a document metadata table.
- Maintain a processed document data table.
- Maintain processing history and error logs.
- Maintain relationships between document metadata and extracted information.
- Implement transaction management to prevent partial persistence.
- Support upsert operations where applicable.
- Ensure duplicate Kafka events do not create duplicate records.

### FR-05: Database-Driven API Generation

The API layer should be driven by database configuration and metadata.

Requirements:

- Store API definitions and configurations in database tables.
- API runtime should retrieve API metadata from the database.
- Support dynamically configured endpoints.
- Define request parameters, response structures, HTTP methods, and database mappings.
- Map API requests to database queries or stored procedures.
- Generate API responses based on database records.
- Support pagination, sorting, filtering, and search.
- Support API versioning.
- Validate incoming requests against database-defined schemas.
- Provide appropriate error responses.

**Suggested API configuration tables:**

1. API_DEFINITION
   - API_ID
   - API_NAME
   - API_PATH
   - HTTP_METHOD
   - API_VERSION
   - STATUS

2. API_REQUEST_MAPPING
   - MAPPING_ID
   - API_ID
   - PARAMETER_NAME
   - PARAMETER_TYPE
   - IS_REQUIRED

3. API_RESPONSE_MAPPING
   - MAPPING_ID
   - API_ID
   - RESPONSE_FIELD
   - DATABASE_COLUMN
   - DATA_TYPE

4. API_DATABASE_MAPPING
   - MAPPING_ID
   - API_ID
   - TABLE_NAME
   - STORED_PROCEDURE
   - QUERY_CONFIGURATION

5. API_SCHEMA
   - SCHEMA_ID
   - API_ID
   - REQUEST_SCHEMA
   - RESPONSE_SCHEMA

### FR-06: Document Retrieval API

Expose database-driven APIs to retrieve processed document information.

Example endpoints:

- GET /api/documents
- GET /api/documents/{documentId}
- GET /api/documents/{documentId}/status
- GET /api/documents/{documentId}/content

The actual endpoint definitions and mappings should be maintained in the database and loaded by the API runtime.

### FR-07: Error Handling and Retry

- Implement Kafka consumer retry mechanisms.
- Configure retry limits.
- Publish permanently failed messages to a Dead Letter Topic (DLT).
- Capture document processing exceptions.
- Maintain error details and timestamps in the database.
- Support reprocessing of failed documents.
- Ensure database failures do not result in lost processing events.

### FR-08: Monitoring and Logging

- Track Kafka consumer lag.
- Monitor document processing throughput.
- Track processing failures and retry counts.
- Log document ID and correlation ID across services.
- Monitor database persistence failures.
- Provide operational visibility into document processing statuses.

## 4. Non-Functional Requirements

### Scalability
- Support multiple Kafka consumers.
- Allow horizontal scaling of document processing services.
- Support partition-based parallel processing.

### Reliability
- Provide at-least-once Kafka delivery.
- Implement idempotent database persistence.
- Support retry and dead-letter handling.

### Performance
- Process documents asynchronously.
- Avoid blocking the document upload request while processing.
- Optimize database queries and indexes.
- Support configurable consumer batch sizes.

### Security
- Secure APIs using OAuth 2.0 / JWT.
- Implement role-based access control.
- Encrypt documents at rest and data in transit.
- Protect database credentials and Kafka credentials.

### Maintainability
- Keep document processing logic separate from API runtime logic.
- Maintain versioned API configurations.
- Support configuration-driven document processing rules.

## 5. Suggested Database Structure

| Table | Purpose |
|---|---|
| DOCUMENT_METADATA | Original document metadata |
| DOCUMENT_DATA | Extracted and processed data |
| DOCUMENT_PROCESSING_HISTORY | Processing lifecycle |
| DOCUMENT_PROCESSING_ERRORS | Error and retry details |
| API_DEFINITION | Dynamic API configurations |
| API_REQUEST_MAPPING | Request parameter definitions |
| API_RESPONSE_MAPPING | Response field definitions |
| API_DATABASE_MAPPING | Database query/table mappings |
| API_SCHEMA | Request and response schemas |

## 6. Suggested Kafka Topics

| Topic | Purpose |
|---|---|
| document.received | New document events |
| document.processing | Document processing requests |
| document.processed | Successfully processed documents |
| document.failed | Failed document events |
| document.retry | Retry processing events |
| document.dlq | Permanently failed events |

## 7. End-to-End Acceptance Criteria

1. A document can be uploaded successfully.
2. A unique document ID is generated.
3. A Kafka event is published successfully.
4. A Kafka consumer receives and processes the event.
5. The document content is extracted and validated.
6. Processed information is persisted into the database.
7. Processing status is updated correctly.
8. Duplicate events do not create duplicate database records.
9. Failed documents can be retried.
10. Permanently failed messages are routed to the dead-letter topic.
11. API definitions can be configured and maintained in the database.
12. The API runtime can load API definitions from the database.
13. A newly configured API can expose the corresponding database data without requiring a new hardcoded endpoint implementation.
14. API responses support validation, pagination, and error handling.
15. End-to-end processing can be traced using document ID and correlation ID.

## 8. Deliverables

- Document ingestion service
- Kafka producer implementation
- Kafka consumer and processing service
- Document parsing and transformation module
- Database schema and migration scripts
- API configuration database schema
- Dynamic API runtime
- Document retrieval APIs
- Retry and dead-letter implementation
- Logging and monitoring
- Unit, integration, and end-to-end tests
- Technical architecture and deployment documentation

## 9. Definition of Done

The solution is considered complete when a document can be ingested, published to Kafka, processed asynchronously, persisted into the database, and retrieved through an API whose configuration is maintained in the database.