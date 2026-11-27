# React Frontend Specification – Document Processing System

**Version:** `1.0.0-draft`  
**Primary Stack:** React + TypeScript, TanStack Query, Redux Toolkit (selective), MUI, React Hook Form, Axios  
**Scope:** Frontend architecture, UX flows, API integration contracts, state management, and non-functional standards.

## 1) Purpose and Scope

This document defines the frontend design and implementation requirements for a document processing UI where ingestion is synchronous (`202 Accepted`) and processing is asynchronous (Kafka + backend consumers).

Out of scope:
- Kafka producer/consumer implementation
- Database schema implementation
- Backend business rules

## 2) Frontend Architecture

## 2.1 Layered Structure

- `app/` – application bootstrapping, providers, route shell
- `pages/` – route-level containers
- `features/documents/` – document-focused modules
- `components/` – reusable presentation components
- `api/` – centralized HTTP client and endpoint wrappers
- `state/` – minimal client-only global state (Redux Toolkit)
- `hooks/` – reusable query/mutation and utility hooks
- `types/` – shared TypeScript contracts
- `utils/` – formatters, validators, mappers

## 2.2 Runtime Flow

1. User uploads document from UI.
2. Frontend POSTs to `/api/documents`.
3. Backend returns document ID and status acknowledgment.
4. Frontend shows optimistic acknowledgment and document ID immediately.
5. Dashboard/detail views poll status endpoint(s) until terminal state.
6. UI fetches processed content once status reaches `PROCESSED`.

## 3) Routing and Pages

Recommended route map:

- `/documents/upload` → Upload page
- `/documents` → Dashboard page
- `/documents/:documentId` → Details page
- `/documents/:documentId/content` → Processed content viewer

Optional guarded routes:
- `/admin/api-config` (if frontend includes dynamic API config admin)

## 4) Functional Requirements Mapping

## 4.1 FR-UI-01: Document Upload

UI behaviors:
- Accept `.pdf`, `.docx`, `.json` (from backend-configurable allowlist).
- Client-side validation:
  - file extension/MIME
  - max file size
- Show selected file name and size.
- Show upload progress (`0-100%`) using Axios progress callbacks.
- Disable submit while upload in progress to prevent duplicates.
- Show success/failure toast/snackbar.

## 4.2 FR-UI-02: Processing Initiation

- Use mutation `uploadDocument` against `POST /api/documents`.
- Render returned `documentId` and `status`.
- Show note: processing is asynchronous.
- Do not block on processing completion.

## 4.3 FR-UI-03: Dashboard

Table columns:
- Document ID
- Document Name
- Document Type
- Uploaded Date
- Processing Status
- Last Updated
- Actions

Features:
- Search by ID/name.
- Filter by type/status.
- Sort by uploaded date.
- Pagination.
- Manual refresh button.

## 4.4 FR-UI-04: Processing Status

Supported statuses:
- `RECEIVED`
- `PENDING`
- `PROCESSING`
- `PROCESSED`
- `FAILED`
- `RETRY_PENDING`

Behavior:
- Status chip color mapping by state.
- Polling with TanStack Query (`refetchInterval`) while non-terminal states.
- Terminal states: `PROCESSED`, `FAILED`.
- Display processing errors when present.
- Show retry action for authorized users.

## 4.5 FR-UI-05: Details Page

Displays:
- Document metadata
- Current status
- Processing history timeline
- Extracted data summary
- Error details
- Created/updated timestamps

## 4.6 FR-UI-06: Processed Data Viewer

- Render JSON in structured tree view.
- Expand/collapse nested nodes.
- Copy to clipboard and download JSON where authorized.

## 4.7 FR-UI-07: Error Handling

- Standard API error mapper to user-facing messages.
- Network failure states with retry CTA.
- Loading and empty states per view.
- Auth expiry handling (`401`) with session expiration flow.

## 5) API Integration Contract

Frontend shall communicate only through backend REST APIs.

| Operation | Method | Endpoint |
|---|---|---|
| Upload document | POST | `/api/documents` |
| List documents | GET | `/api/documents` |
| Get details | GET | `/api/documents/{id}` |
| Get status | GET | `/api/documents/{id}/status` |
| Get content | GET | `/api/documents/{id}/content` |
| Retry processing | POST | `/api/documents/{id}/retry` |

## 5.1 Request/Response Types

```ts
export type ProcessingStatus =
  | 'RECEIVED'
  | 'PENDING'
  | 'PROCESSING'
  | 'PROCESSED'
  | 'FAILED'
  | 'RETRY_PENDING';

export interface UploadDocumentResponse {
  documentId: string;
  status: ProcessingStatus;
  uploadedAt: string;
  fileLocation?: string;
  correlationId?: string;
}

export interface DocumentListItem {
  documentId: string;
  documentName: string;
  documentType: string;
  uploadedDate: string;
  processingStatus: ProcessingStatus;
  lastUpdated: string;
}

export interface DocumentDetails extends DocumentListItem {
  sourceSystem?: string;
  metadata?: Record<string, unknown>;
  processingHistory?: Array<{
    status: ProcessingStatus;
    timestamp: string;
    message?: string;
  }>;
  processingErrors?: Array<{
    code: string;
    message: string;
    timestamp: string;
  }>;
}

export interface DocumentContentResponse {
  documentId: string;
  content: Record<string, unknown>;
  schemaVersion?: string;
}
```

## 5.2 Query Parameters for Dashboard

```ts
export interface DocumentListQuery {
  page: number;
  pageSize: number;
  search?: string;
  documentType?: string;
  processingStatus?: ProcessingStatus;
  sortBy?: 'uploadedDate';
  sortOrder?: 'asc' | 'desc';
}
```

## 6) State Management Strategy

## 6.1 TanStack Query (Server State)

Use query keys:
- `['documents', query]`
- `['document', documentId]`
- `['document-status', documentId]`
- `['document-content', documentId]`

Mutations:
- `uploadDocument`
- `retryDocument`

Invalidation rules:
- After upload success: invalidate `['documents']`
- After retry success: invalidate `['documents']`, `['document', id]`, `['document-status', id]`

Polling strategy:
- Status query polling every `3-5s` for non-terminal states.
- Disable polling at terminal states.

## 6.2 Redux Toolkit (Client State Only)

Use Redux only for client-centric state, not duplicated server data:
- auth/session UI flags
- layout preferences
- global notification queue (if not using Query + context)

## 7) Component Design

Recommended components:
- `DocumentUpload`
- `DocumentDashboard`
- `DocumentTable`
- `DocumentFilters`
- `DocumentStatus`
- `DocumentDetails`
- `ProcessingHistory`
- `ExtractedDataViewer`
- `ErrorDetails`
- `ConfirmationDialog`

Container/presentation split:
- Container: data fetching, mutations, permissions
- Presentation: pure UI rendering

## 8) UX and Interaction Requirements

## 8.1 Upload UX
- Drag/drop + browse.
- Immediate validation errors.
- Progress bar + cancel option (optional).

## 8.2 Dashboard UX
- Sticky filters for large lists.
- Action menu per row: View, View Content, Retry (if eligible).
- Empty state with CTA to upload.

## 8.3 Details UX
- Summary card + tabs:
  - Metadata
  - Processing History
  - Extracted Data
  - Errors

## 9) Error Handling and Notifications

Centralized error parser:

```ts
export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: unknown;
    correlationId?: string;
    timestamp?: string;
  };
}
```

Message policy:
- Display concise user message.
- Log technical details for diagnostics.
- Include correlation ID in expandable error panel where available.

## 10) Accessibility and Responsive Design

- WCAG-aligned labels, focus order, keyboard support.
- ARIA for status chips and progress updates.
- Responsive breakpoints for table -> card layout on smaller screens.
- Color status indicators must include text/icon semantics (not color-only).

## 11) Security Constraints

- Attach OAuth2/JWT bearer token in centralized Axios interceptor.
- Handle `401/403` globally.
- Avoid exposing backend internals (Kafka topics, DB schema internals) in UI.
- Restrict retry/download actions via permission checks.

## 12) Performance Requirements

- Virtualized table rendering for large datasets.
- Debounced search input (`250-400ms`).
- Memoized row renderers where needed.
- Lazy-load heavy views/components (e.g., JSON viewer).

## 13) Environment and Configuration

Environment variables:
- `VITE_API_BASE_URL` (or equivalent)
- `VITE_POLL_INTERVAL_MS`
- `VITE_MAX_UPLOAD_SIZE_MB`
- `VITE_ALLOWED_FILE_TYPES`

All environment values are read at build time and surfaced via typed config module.

## 14) Testing Strategy

Unit tests:
- validators (file type/size)
- status badge mapping
- error parser
- query key builders

Integration tests:
- upload mutation with mocked API
- dashboard filters/sort/pagination behavior
- detail + status polling behavior

E2E tests:
- Upload -> Dashboard visibility
- Async status progression
- Processed content render
- Failed flow + retry

## 15) Acceptance Criteria Traceability

1. Upload supported files: Sections 4.1, 8.1
2. Receive document ID: Sections 4.2, 5.1
3. Dashboard list view: Sections 4.3, 8.2
4. Status updates without full refresh: Sections 4.4, 6.1
5. Metadata/history view: Sections 4.5, 8.3
6. Extracted data view: Sections 4.6, 8.3
7. Meaningful errors: Sections 4.7, 9
8. Retry failed docs: Sections 4.4, 5, 11
9. REST-only backend communication: Section 5
10. Frontend independent from Kafka/DB internals: Sections 1, 11

## 16) Definition of Done

The frontend is done when users can:
- Upload supported documents with client-side validation.
- Receive immediate asynchronous acknowledgment with `documentId`.
- Track processing state from dashboard/details without hard refresh.
- View metadata, history, extracted data, and errors.
- Retry failed documents (if authorized).
- Interact only via backend REST APIs with robust UX, accessibility, and test coverage.