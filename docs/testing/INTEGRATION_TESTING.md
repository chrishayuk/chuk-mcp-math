# Integration Testing Patterns

## Overview

Integration testing verifies that different modules, components, or services work correctly together. This document covers patterns for effective integration testing.

## Core Principles

### Integration Scope
- Test interactions between modules
- Verify data flow through the system
- Validate external service integrations
- Test complete workflows
- Ensure API contracts are met

### Integration vs Unit Tests
| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Scope | Single function/class | Multiple components |
| Dependencies | Mocked | Real or test doubles |
| Speed | Fast (ms) | Slower (seconds) |
| Isolation | Complete | Partial |
| Purpose | Logic correctness | Component interaction |

## Integration Test Organization

### Directory Structure
```
tests/integration/
├── workflows/           # End-to-end workflow tests
├── api/                # API integration tests
├── data_flow/          # Data pipeline tests
├── external/           # External service tests
└── conftest.py         # Shared integration fixtures
```

## Testing Patterns

### Module Integration Testing
```python
@pytest.mark.integration
class TestModuleIntegration:
    """Test integration between modules."""
    
    @pytest.mark.asyncio
    async def test_data_pipeline(self):
        """Test data flows correctly through modules."""
        # Input module
        raw_data = await data_loader.load("input.csv")
        
        # Processing module
        processed = await processor.transform(raw_data)
        
        # Validation module
        validated = await validator.check(processed)
        
        # Output module
        result = await writer.save(validated)
        
        # Verify complete pipeline
        assert result.status == "success"
        assert result.records_processed == len(raw_data)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error handling across modules."""
        with pytest.raises(ValidationError) as exc:
            bad_data = {"invalid": "data"}
            await pipeline.process(bad_data)
        
        # Verify error contains context from all modules
        assert "loader" in str(exc.value)
        assert "processor" in str(exc.value)
```

### API Integration Testing
```python
@pytest.mark.integration
class TestAPIIntegration:
    """Test API endpoints integration."""
    
    @pytest.fixture
    async def client(self):
        """Provide test client."""
        app = create_app(test_mode=True)
        async with TestClient(app) as client:
            yield client
    
    async def test_complete_workflow(self, client):
        """Test complete API workflow."""
        # Create resource
        create_response = await client.post("/api/resources", 
            json={"name": "test", "value": 42})
        assert create_response.status_code == 201
        resource_id = create_response.json()["id"]
        
        # Read resource
        get_response = await client.get(f"/api/resources/{resource_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "test"
        
        # Update resource
        update_response = await client.put(f"/api/resources/{resource_id}",
            json={"value": 100})
        assert update_response.status_code == 200
        
        # Delete resource
        delete_response = await client.delete(f"/api/resources/{resource_id}")
        assert delete_response.status_code == 204
        
        # Verify deletion
        get_response = await client.get(f"/api/resources/{resource_id}")
        assert get_response.status_code == 404
```

### Database Integration Testing
```python
@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database operations integration."""
    
    @pytest.fixture
    async def db(self):
        """Provide test database."""
        # Use test database
        db = await create_test_database()
        await db.migrate()
        yield db
        await db.cleanup()
    
    async def test_transaction_rollback(self, db):
        """Test transaction rollback on error."""
        async with db.transaction() as tx:
            # Insert valid data
            user = await tx.users.create(name="test")
            
            # This should fail and rollback everything
            with pytest.raises(IntegrityError):
                await tx.users.create(id=user.id, name="duplicate")
        
        # Verify rollback
        users = await db.users.all()
        assert len(users) == 0
    
    async def test_cascade_operations(self, db):
        """Test cascade deletes work correctly."""
        # Create parent with children
        parent = await db.parents.create(name="parent")
        child1 = await db.children.create(parent_id=parent.id, name="child1")
        child2 = await db.children.create(parent_id=parent.id, name="child2")
        
        # Delete parent
        await db.parents.delete(parent.id)
        
        # Verify children are deleted
        children = await db.children.filter(parent_id=parent.id)
        assert len(children) == 0
```

### External Service Integration
```python
@pytest.mark.integration
@pytest.mark.external
class TestExternalServices:
    """Test integration with external services."""
    
    @pytest.fixture
    def mock_service(self):
        """Provide mock external service."""
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, "https://api.example.com/data",
                     json={"status": "ok"}, status=200)
            yield rsps
    
    async def test_external_api_integration(self, mock_service):
        """Test integration with external API."""
        client = ExternalAPIClient()
        result = await client.fetch_data()
        
        assert result["status"] == "ok"
        assert len(mock_service.calls) == 1
    
    @pytest.mark.skip(reason="Requires real API access")
    async def test_real_api_integration(self):
        """Test against real external API."""
        # Only run in specific environments
        client = ExternalAPIClient(use_real_api=True)
        result = await client.fetch_data()
        
        assert result is not None
        assert "data" in result
```

## Test Data Management

### Test Fixtures
```python
@pytest.fixture(scope="module")
async def test_data():
    """Provide test data for integration tests."""
    data = await load_test_fixtures("integration_test_data.json")
    yield data
    await cleanup_test_data(data)

@pytest.fixture
async def populated_db(db):
    """Provide database with test data."""
    await db.seed_from_fixtures("test_fixtures.sql")
    yield db
    await db.truncate_all()
```

### Test Containers
```python
import testcontainers

@pytest.fixture(scope="session")
async def postgres_container():
    """Provide PostgreSQL test container."""
    with testcontainers.postgres.PostgresContainer("postgres:14") as postgres:
        yield postgres.get_connection_url()

@pytest.fixture(scope="session")
async def redis_container():
    """Provide Redis test container."""
    with testcontainers.redis.RedisContainer("redis:7") as redis:
        yield redis.get_connection_url()
```

## Performance Considerations

### Optimization Strategies
```python
@pytest.mark.integration
class TestOptimizedIntegration:
    """Optimized integration tests."""
    
    @pytest.fixture(scope="class")
    async def shared_setup(self):
        """Share expensive setup across tests."""
        # Expensive setup done once per class
        db = await create_test_database()
        await db.seed_large_dataset()
        yield db
        await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_with_shared_data(self, shared_setup):
        """Test using shared setup."""
        result = await process_data(shared_setup)
        assert result is not None
```

### Parallel Execution
```python
# pytest.ini
[tool:pytest]
markers =
    integration: Integration tests
    parallel_safe: Safe for parallel execution

# Run integration tests in parallel
# pytest -m "integration and parallel_safe" -n auto
```

## Contract Testing

### API Contract Testing
```python
@pytest.mark.integration
class TestAPIContracts:
    """Test API contracts."""
    
    async def test_response_schema(self, client):
        """Test API response matches schema."""
        response = await client.get("/api/users/1")
        
        # Validate against JSON schema
        schema = load_schema("user_response.json")
        validate(response.json(), schema)
    
    async def test_backwards_compatibility(self, client):
        """Test API maintains backwards compatibility."""
        # Test v1 endpoint still works
        v1_response = await client.get("/api/v1/users")
        assert v1_response.status_code == 200
        
        # Test v2 endpoint with new features
        v2_response = await client.get("/api/v2/users")
        assert v2_response.status_code == 200
        
        # Verify v1 fields exist in v2
        v1_fields = set(v1_response.json()[0].keys())
        v2_fields = set(v2_response.json()[0].keys())
        assert v1_fields.issubset(v2_fields)
```

## Best Practices

### DO's
✅ Test realistic scenarios  
✅ Use test databases/containers  
✅ Test error scenarios  
✅ Verify data consistency  
✅ Test transaction boundaries  
✅ Use appropriate timeouts  
✅ Clean up test data  
✅ Test API contracts  

### DON'Ts
❌ Don't test against production  
❌ Don't skip cleanup  
❌ Don't ignore flaky tests  
❌ Don't hardcode test data  
❌ Don't test implementation details  
❌ Don't make tests order-dependent  
❌ Don't use real external services (without flags)  
❌ Don't share state between tests  

## Environment Setup

### Configuration
```python
# conftest.py for integration tests
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test_db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/1"
    
    yield
    
    # Cleanup
    os.environ.pop("ENVIRONMENT", None)
```

### Test Markers
```python
# Mark integration tests
@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration():
    pass

# Skip if external service unavailable
@pytest.mark.skipif(
    not os.environ.get("EXTERNAL_API_KEY"),
    reason="External API key not configured"
)
async def test_external_service():
    pass
```

## Related Documentation
- [Unit Testing](./UNIT_TESTING.md)
- [Performance Testing](./PERFORMANCE_TESTING.md)
- [Test Fundamentals](./TEST_FUNDAMENTALS.md)
- [Testing Index](./TESTING.md)