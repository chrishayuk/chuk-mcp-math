#!/usr/bin/env python3
# chuk_mcp_functions/__init__.py
"""
Chuk MCP Functions - Comprehensive Function Library for AI Models

A modular collection of MCP-compatible functions designed specifically for AI model execution.
Includes mathematical operations, data processing, utilities, and more.

Key Features:
- Full MCP compliance with resource and tool specifications
- Smart caching and performance optimization
- Comprehensive error handling and validation
- Local and remote execution support
- Streaming capabilities where appropriate
- Rich documentation and examples for AI understanding

Modules:
- math: Complete mathematical operations library
- data: Data processing and manipulation functions
- text: String and text processing utilities
- datetime: Date and time operations
- file: File system operations
- network: Network and API utilities
- conversion: Unit and format conversions
"""

from typing import Dict, List, Optional, Any
import logging

# Import core MCP functionality
from .mcp_pydantic_base import McpPydanticBase, Field, ValidationError
from .mcp_decorator import (
    mcp_function, 
    MCPFunctionSpec,
    ExecutionMode,
    CacheStrategy,
    ResourceLevel,
    StreamingMode,
    get_mcp_functions,
    get_function_by_name,
    export_function_specs,
    print_function_summary
)

# Import all math modules
from .math import (
    # Basic arithmetic
    arithmetic,
    # Advanced math
    # trigonometry,
    # logarithmic,
    # statistical,
    # algebraic,
    # financial,
    # geometric,
    # combinatorial,
    # constants
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Chuk MCP Functions"
__description__ = "Comprehensive MCP function library for AI models"

# Configure logging
logger = logging.getLogger(__name__)

def get_all_functions() -> Dict[str, MCPFunctionSpec]:
    """Get all registered MCP functions across all modules."""
    return get_mcp_functions()

def get_functions_by_category(category: str) -> Dict[str, MCPFunctionSpec]:
    """Get functions filtered by category."""
    all_funcs = get_mcp_functions()
    return {
        name: spec for name, spec in all_funcs.items() 
        if spec.category == category
    }

def get_functions_by_namespace(namespace: str) -> Dict[str, MCPFunctionSpec]:
    """Get functions filtered by namespace."""
    return get_mcp_functions(namespace)

def get_execution_stats() -> Dict[str, Any]:
    """Get comprehensive execution statistics across all functions."""
    all_funcs = get_mcp_functions()
    
    total_functions = len(all_funcs)
    local_count = sum(1 for spec in all_funcs.values() if spec.supports_local_execution())
    remote_count = sum(1 for spec in all_funcs.values() if spec.supports_remote_execution())
    cached_count = sum(1 for spec in all_funcs.values() if spec.cache_strategy != CacheStrategy.NONE)
    streaming_count = sum(1 for spec in all_funcs.values() if spec.supports_streaming)
    
    # Namespace distribution
    namespaces = {}
    categories = {}
    for spec in all_funcs.values():
        namespaces[spec.namespace] = namespaces.get(spec.namespace, 0) + 1
        categories[spec.category] = categories.get(spec.category, 0) + 1
    
    # Performance metrics
    total_executions = 0
    total_errors = 0
    total_cache_hits = 0
    total_cache_misses = 0
    
    for spec in all_funcs.values():
        if spec._performance_metrics:
            total_executions += spec._performance_metrics.execution_count
            total_errors += spec._performance_metrics.error_count
            total_cache_hits += spec._performance_metrics.cache_hits
            total_cache_misses += spec._performance_metrics.cache_misses
    
    cache_hit_rate = 0.0
    if total_cache_hits + total_cache_misses > 0:
        cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
    
    error_rate = 0.0
    if total_executions > 0:
        error_rate = total_errors / total_executions
    
    return {
        "total_functions": total_functions,
        "execution_modes": {
            "local_capable": local_count,
            "remote_capable": remote_count,
            "both_capable": sum(1 for spec in all_funcs.values() 
                               if spec.supports_local_execution() and spec.supports_remote_execution())
        },
        "features": {
            "cached_functions": cached_count,
            "streaming_functions": streaming_count,
            "workflow_compatible": sum(1 for spec in all_funcs.values() if spec.workflow_compatible)
        },
        "distribution": {
            "namespaces": namespaces,
            "categories": categories
        },
        "performance": {
            "total_executions": total_executions,
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses
        }
    }

def print_comprehensive_summary():
    """Print a comprehensive summary of all registered functions."""
    stats = get_execution_stats()
    
    print("🚀 Chuk MCP Functions - Comprehensive Summary")
    print("=" * 50)
    print(f"📊 Total Functions: {stats['total_functions']}")
    print(f"📦 Local Executable: {stats['execution_modes']['local_capable']}")
    print(f"🛠️  Remote Callable: {stats['execution_modes']['remote_capable']}")
    print(f"🔄 Dual Mode: {stats['execution_modes']['both_capable']}")
    print(f"💾 Cached: {stats['features']['cached_functions']}")
    print(f"🌊 Streaming: {stats['features']['streaming_functions']}")
    print(f"🔗 Workflow Ready: {stats['features']['workflow_compatible']}")
    print()
    
    print("📁 By Namespace:")
    for namespace, count in sorted(stats['distribution']['namespaces'].items()):
        print(f"   • {namespace}: {count} functions")
    print()
    
    print("🏷️  By Category:")
    for category, count in sorted(stats['distribution']['categories'].items()):
        print(f"   • {category}: {count} functions")
    print()
    
    if stats['performance']['total_executions'] > 0:
        print("⚡ Performance Metrics:")
        print(f"   • Total Executions: {stats['performance']['total_executions']:,}")
        print(f"   • Error Rate: {stats['performance']['error_rate']:.2%}")
        print(f"   • Cache Hit Rate: {stats['performance']['cache_hit_rate']:.2%}")
        print()

def export_all_specs(filename: str = "mcp_functions_complete.json"):
    """Export all function specifications to a JSON file."""
    export_function_specs(filename)
    print(f"📤 Exported all function specifications to {filename}")

def clear_all_caches():
    """Clear all function caches."""
    cleared_count = 0
    for spec in get_mcp_functions().values():
        if spec._cache_backend:
            spec._cache_backend.clear()
            cleared_count += 1
    
    print(f"🗑️  Cleared {cleared_count} function caches")

# Export main components
__all__ = [
    # Core MCP components
    'McpPydanticBase', 'Field', 'ValidationError',
    'mcp_function', 'MCPFunctionSpec',
    'ExecutionMode', 'CacheStrategy', 'ResourceLevel', 'StreamingMode',
    
    # Function management
    'get_mcp_functions', 'get_function_by_name', 'get_all_functions',
    'get_functions_by_category', 'get_functions_by_namespace',
    
    # Statistics and management
    'get_execution_stats', 'print_function_summary', 'print_comprehensive_summary',
    'export_function_specs', 'export_all_specs', 'clear_all_caches',
    
    # Math modules
    'arithmetic'#, 'trigonometry', 'logarithmic', 'statistical', 
    #'algebraic', 'financial', 'geometric', 'combinatorial', 'constants',
    
    # Package info
    '__version__', '__author__', '__description__'
]

# Initialize logging for the package
def setup_logging(level: str = "INFO"):
    """Setup package-wide logging."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Chuk MCP Functions v{__version__} initialized")
    logger.info(f"Loaded {len(get_mcp_functions())} functions")

# Auto-setup logging at import
setup_logging()

if __name__ == "__main__":
    print_comprehensive_summary()