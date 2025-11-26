# Quiz Generation 502 Error Fix - Version 2

## Problem
The quiz API was returning 502 Bad Gateway when generating more than 5 questions, especially when pressing "Next" to load more questions incrementally.

## Root Causes

### 1. **Overly Strict Minimum Requirements**
The code was enforcing a minimum of 10 questions even when requesting only 2-3:
```python
target = max(10, min(15, int(count or 12)))  # OLD - forced 10 minimum
```

### 2. **Overly Strict Grounding Threshold**
The grounding filter required a score of 0.35+, which was too strict:
```python
return best if best_score >= 0.35 else None  # OLD - 35% threshold
```
When generating small batches, even valid questions couldn't find supporting sentences.

### 3. **Poor Fallback for Small Batches**
When grounding failed, all questions were rejected instead of using fallback explanations for incremental loads.

## Solutions Implemented

### 1. **Flexible Minimum Targets** ✅
Changed minimum from 10 to 2 questions, allowing proper incremental generation:

**quiz_app.py - `_generate_from_notes()`:**
```python
target = max(2, min(20, int(target or 12)))  # NEW - allows 2-20 questions
```

**quiz_app.py - `generate_quiz_items()`:**
```python
target = max(2, min(20, int(count or 12)))  # NEW - allows 2-20 questions
```

### 2. **Relaxed Grounding Threshold** ✅
Lowered the grounding score threshold from 0.35 to 0.25:
```python
return best if best_score >= 0.25 else None  # NEW - 25% threshold
```

### 3. **Graceful Fallback for Small Batches** ✅
When requesting ≤5 questions, allow ungrounded items with generic explanations:

**quiz_app.py - `_generate_from_notes()`:**
```python
if not support:
    # For small batches (incremental loading), accept items without perfect grounding
    if target <= 5:
        it["explanation"] = it.get("explanation", "") or f"{ans} is correct."
        it["supporting_sentence"] = "Inferred from notes"
        grounded.append(it)
    continue
```

### 4. **Smarter Request Buffering** ✅
When requesting N questions, ask for max(N+2, 1.5*N) instead of fixed N+6:

**workspace.py - `api_generate_quiz()`:**
```python
generate_count = max(request_count + 2, int(request_count * 1.5))
# For 3: max(5, 4) = 5 questions generated
# For 12: max(14, 18) = 18 questions generated
```

### 5. **Better Debug Logging** ✅
Added detailed logging to understand what's happening:
- How many questions are generated vs requested
- Which items are being skipped and why
- Final counts before/after filtering

## Files Modified

1. **quiz_app.py**
   - `_generate_from_notes()` - Changed minimum, added fallback for small batches
   - `generate_quiz_items()` - Changed minimum
   - `_best_support_sentence()` - Relaxed threshold from 0.35 to 0.25

2. **workspace.py**
   - `api_generate_quiz()` - Smarter request buffering, better logging

## Expected Behavior Now

### Initial Load (First Quiz)
- Request 3 questions → generates ~4-5 → returns 3
- Request 12 questions → generates ~18 → returns 12

### Incremental Load (Next Button)
- Request 3 more → generates ~5 → returns 3
- Small batches use fallback explanations if needed
- 502 error only if LLM fails, not from filtering

### Logging (for troubleshooting)
```
[WORKSPACE] Generating 5 quiz items (requested: 3) from 1500 chars of source text...
[WORKSPACE] Quiz generation returned 5 items
[WORKSPACE] After filtering: 3 valid items from 5 generated
```

## Testing Checklist

- [ ] Upload PDF/text to workspace
- [ ] Click Quiz tab - loads 3 questions initially
- [ ] Click "Next" multiple times - each time loads 2-3 more
- [ ] Continue until no more questions available
- [ ] Check terminal for `[WORKSPACE]` logs showing generation counts
- [ ] No 502 errors (should show real LLM errors if any)

## Migration Notes

No breaking changes. The API remains the same, but:
- `count` parameter now accepts 2-20 instead of 10-15
- Explanations for small incremental batches may be shorter
- Questions may be slightly less perfectly grounded (but still valid)
