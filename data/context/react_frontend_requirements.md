# React Frontend Requirements – Document Processing System

## 1. Technology Stack

- React with TypeScript
- Redux Toolkit for application state management, if shared state is required
- TanStack Query for API calls, caching, and server-state management
- Material UI (MUI) for UI components
- React Hook Form for form handling
- Axios or a centralized API client for HTTP communication

## 2. Functional Requirements

### FR-UI-01: Document Upload
- Provide an interface to upload documents.
- Support PDF, DOCX, and JSON formats as configured by the backend.
- Validate file type and file size before submission.
- Display selected document name and file size.
- Show upload progress.
- Display success or failure notifications.
- Prevent duplicate submissions while an upload is in progress.

### FR-UI-02: Document Processing Initiation
- Submit the document to the backend ingestion API.
- Receive a unique document ID and processing acknowledgment.
- Display the document ID.
- Inform the user that processing happens asynchronously through Kafka.
- Do not keep the upload request waiting for document processing to finish.

### FR-UI-03: Document Processing Dashboard
Display uploaded documents in a tabular view.

Required columns:
- Document ID
- Document Name
- Document Type
- Uploaded Date
- Processing Status
- Last Updated
- Actions

Provide:
- Search by document ID and name.
- Filter by document type.
- Filter by processing status.
- Sort by uploaded date.
- Pagination.
- Refresh functionality.

### FR-UI-04: Processing Status
Display real-time or periodically refreshed processing status.

Supported statuses:
- RECEIVED
- PENDING
- PROCESSING
- PROCESSED
- FAILED
- RETRY_PENDING

Requirements:
- Use appropriate status indicators.
- Refresh status using TanStack Query polling or server-sent events/WebSockets.
- Display processing errors when available.
- Allow users to retry failed documents if authorized.

### FR-UI-05: Document Details
Provide a document details page containing:
- Document metadata.
- Processing status.
- Processing history.
- Extracted document information.
- Processing errors.
- Created and updated timestamps.

### FR-UI-06: Processed Document Viewer
- Display extracted document information.
- Support structured JSON data rendering.
- Provide expandable sections for nested data.
- Support copying or downloading extracted data where authorized.

### FR-UI-07: Error Handling
- Display meaningful API error messages.
- Handle network failures.
- Display loading and empty states.
- Handle expired authentication sessions.
- Provide retry actions where applicable.

## 3. API Integration Requirements

The React application shall communicate with the backend through REST APIs.

Example API operations:

| Operation | HTTP Method | Endpoint |
|---|---|---|
| Upload document | POST | /api/documents |
| List documents | GET | /api/documents |
| Get document details | GET | /api/documents/{id} |
| Get processing status | GET | /api/documents/{id}/status |
| Get extracted data | GET | /api/documents/{id}/content |
| Retry processing | POST | /api/documents/{id}/retry |

The frontend shall not communicate directly with Kafka or the database.

The API layer shall abstract Kafka processing and database persistence.

## 4. State Management Requirements

- Use TanStack Query for API server state.
- Use Redux Toolkit for shared client-side application state where necessary.
- Avoid duplicating server data in Redux.
- Centralize API service functions.
- Configure query caching and invalidation.
- Implement polling for processing status.
- Invalidate relevant queries after document upload or retry.

## 5. UI Component Structure

Suggested component organization:

- DocumentUpload
- DocumentDashboard
- DocumentTable
- DocumentFilters
- DocumentStatus
- DocumentDetails
- ProcessingHistory
- ExtractedDataViewer
- ErrorDetails
- ConfirmationDialog

## 6. Non-Functional Requirements

- Responsive UI.
- Accessibility compliance.
- Secure authentication and authorization.
- Lazy loading for larger pages.
- Optimized rendering for large document lists.
- Reusable components.
- Centralized error handling.
- Unit and integration tests.
- Environment-specific API configurations.

## 7. Acceptance Criteria

1. Users can upload supported document types.
2. Frontend receives a document ID after successful ingestion.
3. Users can see uploaded documents in the dashboard.
4. Processing status is updated without requiring a full page refresh.
5. Users can view document metadata and processing history.
6. Users can view extracted document information after successful processing.
7. Users receive meaningful error messages when processing fails.
8. Authorized users can retry failed documents.
9. All backend communication happens through REST APIs.
10. Frontend components remain independent of Kafka and database implementation details.

## 8. Definition of Done

The React application is complete when users can upload documents, track asynchronous processing, view processing results, and retrieve extracted information through backend APIs.