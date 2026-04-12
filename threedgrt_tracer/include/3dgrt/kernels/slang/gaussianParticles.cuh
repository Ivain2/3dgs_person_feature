#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};

struct gaussianParticle_RawParameters_0
{
    float3  position_0;
    float density_0;
    float4  quaternion_0;
    float3  scale_0;
    float padding_0;
};

__device__ gaussianParticle_RawParameters_0 gaussianParticle_RawParameters_x24_syn_dzero_0()
{
    gaussianParticle_RawParameters_0 result_0;
    float3  _S1 = make_float3 (0.0f);
    (&result_0)->position_0 = _S1;
    (&result_0)->density_0 = 0.0f;
    (&result_0)->quaternion_0 = make_float4 (0.0f);
    (&result_0)->scale_0 = _S1;
    (&result_0)->padding_0 = 0.0f;
    return result_0;
}

__device__ Matrix<float, 3, 3>  transforms_rotationMatrixTranspose_0(float4  quaternion_1)
{
    float _S2 = quaternion_1.y;
    float xx_0 = _S2 * _S2;
    float _S3 = quaternion_1.z;
    float yy_0 = _S3 * _S3;
    float _S4 = quaternion_1.w;
    float zz_0 = _S4 * _S4;
    float xy_0 = _S2 * _S3;
    float xz_0 = _S2 * _S4;
    float yz_0 = _S3 * _S4;
    float _S5 = quaternion_1.x;
    float rx_0 = _S5 * _S2;
    float ry_0 = _S5 * _S3;
    float rz_0 = _S5 * _S4;
    return makeMatrix<float, 3, 3> (make_float3 (1.0f - 2.0f * (yy_0 + zz_0), 2.0f * (xy_0 + rz_0), 2.0f * (xz_0 - ry_0)), make_float3 (2.0f * (xy_0 - rz_0), 1.0f - 2.0f * (xx_0 + zz_0), 2.0f * (yz_0 + rx_0)), make_float3 (2.0f * (xz_0 + ry_0), 2.0f * (yz_0 - rx_0), 1.0f - 2.0f * (xx_0 + yy_0)));
}

struct gaussianParticle_Parameters_0
{
    float3  position_1;
    float3  scale_1;
    Matrix<float, 3, 3>  rotationT_0;
    float density_1;
};

struct gaussianParticle_RawParametersBuffer_0
{
    gaussianParticle_RawParameters_0 * _dataPtr_0;
    gaussianParticle_RawParameters_0 * _gradPtr_0;
    bool exclusiveGradient_0;
};

struct gaussianParticle_CommonParameters_0
{
    gaussianParticle_RawParametersBuffer_0 parametersBuffer_0;
};

__device__ gaussianParticle_Parameters_0 particleDensityParameters(uint particleIdx_0, gaussianParticle_CommonParameters_0 commonParameters_0)
{
    gaussianParticle_RawParameters_0 * _S6 = commonParameters_0.parametersBuffer_0._dataPtr_0 + particleIdx_0;
    gaussianParticle_Parameters_0 _S7 = { (*_S6).position_0, (*_S6).scale_0, transforms_rotationMatrixTranspose_0((*_S6).quaternion_0), (*_S6).density_0 };
    return _S7;
}

struct DiffPair_matrixx3Cfloatx2C3x2C3x3E_0
{
    Matrix<float, 3, 3>  primal_0;
    Matrix<float, 3, 3>  differential_0;
};

struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};

__device__ void _d_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * left_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * right_0, float3  dOut_0)
{
    float3  right_d_result_0;
    float _S8 = (*left_0).primal_0.rows[int(0)].x * dOut_0.x;
    Matrix<float, 3, 3>  left_d_result_0;
    *&(((&left_d_result_0)->rows + (int(0)))->x) = (*right_0).primal_0.x * dOut_0.x;
    float sum_0 = _S8 + (*left_0).primal_0.rows[int(1)].x * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->x) = (*right_0).primal_0.x * dOut_0.y;
    float sum_1 = sum_0 + (*left_0).primal_0.rows[int(2)].x * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->x) = (*right_0).primal_0.x * dOut_0.z;
    *&((&right_d_result_0)->x) = sum_1;
    float _S9 = (*left_0).primal_0.rows[int(0)].y * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->y) = (*right_0).primal_0.y * dOut_0.x;
    float sum_2 = _S9 + (*left_0).primal_0.rows[int(1)].y * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->y) = (*right_0).primal_0.y * dOut_0.y;
    float sum_3 = sum_2 + (*left_0).primal_0.rows[int(2)].y * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->y) = (*right_0).primal_0.y * dOut_0.z;
    *&((&right_d_result_0)->y) = sum_3;
    float _S10 = (*left_0).primal_0.rows[int(0)].z * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->z) = (*right_0).primal_0.z * dOut_0.x;
    float sum_4 = _S10 + (*left_0).primal_0.rows[int(1)].z * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->z) = (*right_0).primal_0.z * dOut_0.y;
    float sum_5 = sum_4 + (*left_0).primal_0.rows[int(2)].z * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->z) = (*right_0).primal_0.z * dOut_0.z;
    *&((&right_d_result_0)->z) = sum_5;
    left_0->primal_0 = (*left_0).primal_0;
    left_0->differential_0 = left_d_result_0;
    right_0->primal_0 = (*right_0).primal_0;
    right_0->differential_0 = right_d_result_0;
    return;
}

__device__ float3  mul_0(Matrix<float, 3, 3>  left_1, float3  right_1)
{
    float3  result_1;
    int i_0 = int(0);
    for(;;)
    {
        if(i_0 < int(3))
        {
        }
        else
        {
            break;
        }
        int _S11 = i_0;
        int j_0 = int(0);
        float sum_6 = 0.0f;
        for(;;)
        {
            if(j_0 < int(3))
            {
            }
            else
            {
                break;
            }
            float sum_7 = sum_6 + _slang_vector_get_element(left_1.rows[_S11], j_0) * _slang_vector_get_element(right_1, j_0);
            j_0 = j_0 + int(1);
            sum_6 = sum_7;
        }
        *_slang_vector_get_element_ptr(&result_1, i_0) = sum_6;
        i_0 = i_0 + int(1);
    }
    return result_1;
}

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_1)
{
    DiffPair_float_0 _S12 = *dpx_0;
    float _S13;
    if((*dpx_0).primal_0 > (*dpy_0).primal_0)
    {
        _S13 = dOut_1;
    }
    else
    {
        _S13 = 0.0f;
    }
    dpx_0->primal_0 = _S12.primal_0;
    dpx_0->differential_0 = _S13;
    DiffPair_float_0 _S14 = *dpy_0;
    if((*dpy_0).primal_0 > _S12.primal_0)
    {
        _S13 = dOut_1;
    }
    else
    {
        _S13 = 0.0f;
    }
    dpy_0->primal_0 = _S14.primal_0;
    dpy_0->differential_0 = _S13;
    return;
}

__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_1, float dOut_2)
{
    float _S15 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_1).primal_0)))))) * dOut_2;
    dpx_1->primal_0 = (*dpx_1).primal_0;
    dpx_1->differential_0 = _S15;
    return;
}

__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_3)
{
    float3  x_d_result_0;
    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_3;
    float3  y_d_result_0;
    *&((&y_d_result_0)->x) = (*dpx_2).primal_0.x * dOut_3;
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_3;
    *&((&y_d_result_0)->y) = (*dpx_2).primal_0.y * dOut_3;
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_3;
    *&((&y_d_result_0)->z) = (*dpx_2).primal_0.z * dOut_3;
    dpx_2->primal_0 = (*dpx_2).primal_0;
    dpx_2->differential_0 = x_d_result_0;
    dpy_1->primal_0 = (*dpy_1).primal_0;
    dpy_1->differential_0 = y_d_result_0;
    return;
}

__device__ float dot_0(float3  x_0, float3  y_0)
{
    int i_1 = int(0);
    float result_2 = 0.0f;
    for(;;)
    {
        if(i_1 < int(3))
        {
        }
        else
        {
            break;
        }
        float result_3 = result_2 + _slang_vector_get_element(x_0, i_1) * _slang_vector_get_element(y_0, i_1);
        i_1 = i_1 + int(1);
        result_2 = result_3;
    }
    return result_2;
}

__device__ float length_0(float3  x_1)
{
    return (F32_sqrt((dot_0(x_1, x_1))));
}

__device__ float3  normalize_0(float3  x_2)
{
    return x_2 / make_float3 (length_0(x_2));
}

__device__ void _d_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * a_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * b_0, float3  dOut_4)
{
    float _S16 = dOut_4.y;
    float _S17 = dOut_4.z;
    float _S18 = dOut_4.x;
    float _S19 = (*a_0).primal_0.z * _S16 + - (*a_0).primal_0.y * _S17;
    float _S20 = - (*a_0).primal_0.z * _S18 + (*a_0).primal_0.x * _S17;
    float _S21 = (*a_0).primal_0.y * _S18 + - (*a_0).primal_0.x * _S16;
    float3  _S22 = make_float3 (- (*b_0).primal_0.z * _S16 + (*b_0).primal_0.y * _S17, (*b_0).primal_0.z * _S18 + - (*b_0).primal_0.x * _S17, - (*b_0).primal_0.y * _S18 + (*b_0).primal_0.x * _S16);
    a_0->primal_0 = (*a_0).primal_0;
    a_0->differential_0 = _S22;
    float3  _S23 = make_float3 (_S19, _S20, _S21);
    b_0->primal_0 = (*b_0).primal_0;
    b_0->differential_0 = _S23;
    return;
}

__device__ float3  cross_0(float3  left_2, float3  right_2)
{
    float _S24 = left_2.y;
    float _S25 = right_2.z;
    float _S26 = left_2.z;
    float _S27 = right_2.y;
    float _S28 = right_2.x;
    float _S29 = left_2.x;
    return make_float3 (_S24 * _S25 - _S26 * _S27, _S26 * _S28 - _S29 * _S25, _S29 * _S27 - _S24 * _S28);
}

__device__ void _d_exp_0(DiffPair_float_0 * dpx_3, float dOut_5)
{
    float _S30 = (F32_exp(((*dpx_3).primal_0))) * dOut_5;
    dpx_3->primal_0 = (*dpx_3).primal_0;
    dpx_3->differential_0 = _S30;
    return;
}

__device__ void _d_min_0(DiffPair_float_0 * dpx_4, DiffPair_float_0 * dpy_2, float dOut_6)
{
    DiffPair_float_0 _S31 = *dpx_4;
    float _S32;
    if((*dpx_4).primal_0 < (*dpy_2).primal_0)
    {
        _S32 = dOut_6;
    }
    else
    {
        _S32 = 0.0f;
    }
    dpx_4->primal_0 = _S31.primal_0;
    dpx_4->differential_0 = _S32;
    DiffPair_float_0 _S33 = *dpy_2;
    if((*dpy_2).primal_0 < _S31.primal_0)
    {
        _S32 = dOut_6;
    }
    else
    {
        _S32 = 0.0f;
    }
    dpy_2->primal_0 = _S33.primal_0;
    dpy_2->differential_0 = _S32;
    return;
}

__device__ void _d_mul_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * left_3, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * right_3, float3  dOut_7)
{
    Matrix<float, 3, 3>  right_d_result_1;
    float3  left_d_result_1;
    float _S34 = (*right_3).primal_0.rows[int(0)].x * dOut_7.x;
    *&(((&right_d_result_1)->rows + (int(0)))->x) = (*left_3).primal_0.x * dOut_7.x;
    float sum_8 = _S34 + (*right_3).primal_0.rows[int(0)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(0)))->y) = (*left_3).primal_0.x * dOut_7.y;
    float sum_9 = sum_8 + (*right_3).primal_0.rows[int(0)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(0)))->z) = (*left_3).primal_0.x * dOut_7.z;
    *&((&left_d_result_1)->x) = sum_9;
    float _S35 = (*right_3).primal_0.rows[int(1)].x * dOut_7.x;
    *&(((&right_d_result_1)->rows + (int(1)))->x) = (*left_3).primal_0.y * dOut_7.x;
    float sum_10 = _S35 + (*right_3).primal_0.rows[int(1)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(1)))->y) = (*left_3).primal_0.y * dOut_7.y;
    float sum_11 = sum_10 + (*right_3).primal_0.rows[int(1)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(1)))->z) = (*left_3).primal_0.y * dOut_7.z;
    *&((&left_d_result_1)->y) = sum_11;
    float _S36 = (*right_3).primal_0.rows[int(2)].x * dOut_7.x;
    *&(((&right_d_result_1)->rows + (int(2)))->x) = (*left_3).primal_0.z * dOut_7.x;
    float sum_12 = _S36 + (*right_3).primal_0.rows[int(2)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(2)))->y) = (*left_3).primal_0.z * dOut_7.y;
    float sum_13 = sum_12 + (*right_3).primal_0.rows[int(2)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(2)))->z) = (*left_3).primal_0.z * dOut_7.z;
    *&((&left_d_result_1)->z) = sum_13;
    left_3->primal_0 = (*left_3).primal_0;
    left_3->differential_0 = left_d_result_1;
    right_3->primal_0 = (*right_3).primal_0;
    right_3->differential_0 = right_d_result_1;
    return;
}

__device__ float3  mul_1(float3  left_4, Matrix<float, 3, 3>  right_4)
{
    float3  result_4;
    int j_1 = int(0);
    for(;;)
    {
        if(j_1 < int(3))
        {
        }
        else
        {
            break;
        }
        int i_2 = int(0);
        float sum_14 = 0.0f;
        for(;;)
        {
            if(i_2 < int(3))
            {
            }
            else
            {
                break;
            }
            float sum_15 = sum_14 + _slang_vector_get_element(left_4, i_2) * _slang_vector_get_element(right_4.rows[i_2], j_1);
            i_2 = i_2 + int(1);
            sum_14 = sum_15;
        }
        *_slang_vector_get_element_ptr(&result_4, j_1) = sum_14;
        j_1 = j_1 + int(1);
    }
    return result_4;
}

__device__ bool particleDensityHit(float3  rayOrigin_0, float3  rayDirection_0, gaussianParticle_Parameters_0 parameters_0, float * alpha_0, float * depth_0, bool enableNormal_0, float3  * normal_0)
{
    float3  giscl_0 = make_float3 (1.0f) / parameters_0.scale_1;
    float3  canonicalRayOrigin_0 = giscl_0 * mul_0(parameters_0.rotationT_0, rayOrigin_0 - parameters_0.position_1);
    float3  canonicalRayDirection_0 = normalize_0(giscl_0 * mul_0(parameters_0.rotationT_0, rayDirection_0));
    float3  gcrod_0 = cross_0(canonicalRayDirection_0, canonicalRayOrigin_0);
    float _S37 = dot_0(gcrod_0, gcrod_0);
    float _S38 = (F32_exp((-0.0555555559694767f * _S37 * _S37)));
    *alpha_0 = (F32_min((0.99000000953674316f), (_S38 * parameters_0.density_1)));
    bool acceptHit_0;
    if(_S38 > 0.01130000036209822f)
    {
        acceptHit_0 = *alpha_0 > 0.00392156885936856f;
    }
    else
    {
        acceptHit_0 = false;
    }
    if(acceptHit_0)
    {
        float3  grds_0 = parameters_0.scale_1 * canonicalRayDirection_0 * make_float3 (dot_0(canonicalRayDirection_0, make_float3 (-1.0f) * canonicalRayOrigin_0));
        *depth_0 = (F32_sqrt((dot_0(grds_0, grds_0))));
        if(enableNormal_0)
        {
            float3  surfelNm_0 = make_float3 (0.0f, 0.0f, 1.0f);
            float3  surfelNm_1;
            if(dot_0(surfelNm_0, canonicalRayDirection_0) > 0.0f)
            {
                surfelNm_1 = surfelNm_0 * make_float3 (-1.0f);
            }
            else
            {
                surfelNm_1 = surfelNm_0;
            }
            float3  _S39 = normalize_0(mul_1(surfelNm_1 * parameters_0.scale_1, parameters_0.rotationT_0));
            *normal_0 = _S39;
        }
    }
    return acceptHit_0;
}

__device__ void _d_lerp_0(DiffPair_float_0 * dpx_5, DiffPair_float_0 * dpy_3, DiffPair_float_0 * dps_0, float dOut_8)
{
    float _S40 = (1.0f - (*dps_0).primal_0) * dOut_8;
    dpx_5->primal_0 = (*dpx_5).primal_0;
    dpx_5->differential_0 = _S40;
    DiffPair_float_0 _S41 = *dpy_3;
    float _S42 = (*dps_0).primal_0 * dOut_8;
    dpy_3->primal_0 = (*dpy_3).primal_0;
    dpy_3->differential_0 = _S42;
    float _S43 = (_S41.primal_0 - (*dpx_5).primal_0) * dOut_8;
    dps_0->primal_0 = _S41.primal_0;
    dps_0->differential_0 = _S43;
    return;
}

__device__ void _d_lerp_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_6, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_4, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpz_0, float3  dOut_9)
{
    DiffPair_float_0 left_dp_0;
    (&left_dp_0)->primal_0 = (*dpx_6).primal_0.x;
    (&left_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_0;
    (&middle_dp_0)->primal_0 = (*dpy_4).primal_0.x;
    (&middle_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_0;
    (&right_dp_0)->primal_0 = (*dpz_0).primal_0.x;
    (&right_dp_0)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_0, &middle_dp_0, &right_dp_0, dOut_9.x);
    float3  left_d_result_2;
    *&((&left_d_result_2)->x) = left_dp_0.differential_0;
    float3  middle_d_result_0;
    *&((&middle_d_result_0)->x) = middle_dp_0.differential_0;
    float3  right_d_result_2;
    *&((&right_d_result_2)->x) = right_dp_0.differential_0;
    DiffPair_float_0 left_dp_1;
    (&left_dp_1)->primal_0 = (*dpx_6).primal_0.y;
    (&left_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_1;
    (&middle_dp_1)->primal_0 = (*dpy_4).primal_0.y;
    (&middle_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_1;
    (&right_dp_1)->primal_0 = (*dpz_0).primal_0.y;
    (&right_dp_1)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_1, &middle_dp_1, &right_dp_1, dOut_9.y);
    *&((&left_d_result_2)->y) = left_dp_1.differential_0;
    *&((&middle_d_result_0)->y) = middle_dp_1.differential_0;
    *&((&right_d_result_2)->y) = right_dp_1.differential_0;
    DiffPair_float_0 left_dp_2;
    (&left_dp_2)->primal_0 = (*dpx_6).primal_0.z;
    (&left_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_2;
    (&middle_dp_2)->primal_0 = (*dpy_4).primal_0.z;
    (&middle_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_2;
    (&right_dp_2)->primal_0 = (*dpz_0).primal_0.z;
    (&right_dp_2)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_2, &middle_dp_2, &right_dp_2, dOut_9.z);
    *&((&left_d_result_2)->z) = left_dp_2.differential_0;
    *&((&middle_d_result_0)->z) = middle_dp_2.differential_0;
    *&((&right_d_result_2)->z) = right_dp_2.differential_0;
    dpx_6->primal_0 = (*dpx_6).primal_0;
    dpx_6->differential_0 = left_d_result_2;
    dpy_4->primal_0 = (*dpy_4).primal_0;
    dpy_4->differential_0 = middle_d_result_0;
    dpz_0->primal_0 = (*dpz_0).primal_0;
    dpz_0->differential_0 = right_d_result_2;
    return;
}

__device__ float particleDensityIntegrateHit(float alpha_1, float * transmittance_0, float depth_1, float * integratedDepth_0, bool enableNormal_1, float3  normal_1, float3  * integratedNormal_0)
{
    float _S44 = alpha_1 * *transmittance_0;
    *integratedDepth_0 = *integratedDepth_0 + depth_1 * _S44;
    if(enableNormal_1)
    {
        *integratedNormal_0 = *integratedNormal_0 + normal_1 * make_float3 (_S44);
    }
    *transmittance_0 = *transmittance_0 * (1.0f - alpha_1);
    return _S44;
}

__device__ float particleDensityProcessHitFwdFromBuffer(float3  rayOrigin_1, float3  rayDirection_1, uint particleIdx_1, gaussianParticle_CommonParameters_0 commonParameters_1, float * transmittance_1, float * integratedDepth_1, bool enableNormal_2, float3  * integratedNormal_1)
{
    float depth_2;
    gaussianParticle_RawParameters_0 * _S45 = commonParameters_1.parametersBuffer_0._dataPtr_0 + particleIdx_1;
    float3  _S46 = make_float3 (1.0f);
    float3  surfelNm_2 = make_float3 (0.0f, 0.0f, 1.0f);
    float3  surfelNm_3 = surfelNm_2 * make_float3 (-1.0f);
    for(;;)
    {
        gaussianParticle_RawParameters_0 rawParameters_0 = *_S45;
        Matrix<float, 3, 3>  _S47 = transforms_rotationMatrixTranspose_0((*_S45).quaternion_0);
        float3  giscl_1 = _S46 / rawParameters_0.scale_0;
        float3  canonicalRayOrigin_1 = giscl_1 * mul_0(_S47, rayOrigin_1 - rawParameters_0.position_0);
        float3  canonicalRayDirection_1 = normalize_0(giscl_1 * mul_0(_S47, rayDirection_1));
        float3  gcrod_1 = cross_0(canonicalRayDirection_1, canonicalRayOrigin_1);
        float _S48 = dot_0(gcrod_1, gcrod_1);
        float _S49 = (F32_exp((-0.0555555559694767f * _S48 * _S48)));
        float alpha_2 = (F32_min((0.99000000953674316f), (_S49 * rawParameters_0.density_0)));
        bool acceptHit_1;
        if(_S49 > 0.01130000036209822f)
        {
            acceptHit_1 = alpha_2 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_1 = false;
        }
        float3  normal_2;
        if(acceptHit_1)
        {
            float3  grds_1 = rawParameters_0.scale_0 * canonicalRayDirection_1 * make_float3 (dot_0(canonicalRayDirection_1, make_float3 (-1.0f) * canonicalRayOrigin_1));
            float _S50 = (F32_sqrt((dot_0(grds_1, grds_1))));
            if(enableNormal_2)
            {
                if(dot_0(surfelNm_2, canonicalRayDirection_1) > 0.0f)
                {
                    normal_2 = surfelNm_3;
                }
                else
                {
                    normal_2 = surfelNm_2;
                }
                float3  _S51 = normalize_0(mul_1(normal_2 * rawParameters_0.scale_0, _S47));
                normal_2 = _S51;
            }
            depth_2 = _S50;
        }
        if(acceptHit_1)
        {
            float _S52 = alpha_2 * *transmittance_1;
            *integratedDepth_1 = *integratedDepth_1 + depth_2 * _S52;
            if(enableNormal_2)
            {
                *integratedNormal_1 = *integratedNormal_1 + normal_2 * make_float3 (_S52);
            }
            *transmittance_1 = *transmittance_1 * (1.0f - alpha_2);
            depth_2 = _S52;
            break;
        }
        depth_2 = 0.0f;
        break;
    }
    return depth_2;
}

struct s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0
{
    float _S53;
    float _S54;
    float3  _S55;
    gaussianParticle_RawParameters_0 _S56;
};

__device__ Matrix<float, 3, 3>  s_primal_ctx_transforms_rotationMatrixTranspose_0(float4  dpquaternion_0)
{
    float _S57 = dpquaternion_0.y;
    float xx_1 = _S57 * _S57;
    float _S58 = dpquaternion_0.z;
    float yy_1 = _S58 * _S58;
    float _S59 = dpquaternion_0.w;
    float zz_1 = _S59 * _S59;
    float xy_1 = _S57 * _S58;
    float xz_1 = _S57 * _S59;
    float yz_1 = _S58 * _S59;
    float _S60 = dpquaternion_0.x;
    float rx_1 = _S60 * _S57;
    float ry_1 = _S60 * _S58;
    float rz_1 = _S60 * _S59;
    return makeMatrix<float, 3, 3> (make_float3 (1.0f - 2.0f * (yy_1 + zz_1), 2.0f * (xy_1 + rz_1), 2.0f * (xz_1 - ry_1)), make_float3 (2.0f * (xy_1 - rz_1), 1.0f - 2.0f * (xx_1 + zz_1), 2.0f * (yz_1 + rx_1)), make_float3 (2.0f * (xz_1 + ry_1), 2.0f * (yz_1 - rx_1), 1.0f - 2.0f * (xx_1 + yy_1)));
}

__device__ float3  s_primal_ctx_mul_0(Matrix<float, 3, 3>  _S61, float3  _S62)
{
    return mul_0(_S61, _S62);
}

__device__ float3  s_primal_ctx_cross_0(float3  _S63, float3  _S64)
{
    return cross_0(_S63, _S64);
}

__device__ float s_primal_ctx_dot_0(float3  _S65, float3  _S66)
{
    return dot_0(_S65, _S66);
}

__device__ float s_primal_ctx_exp_0(float _S67)
{
    return (F32_exp((_S67)));
}

__device__ float s_primal_ctx_min_0(float _S68, float _S69)
{
    return (F32_min((_S68), (_S69)));
}

__device__ float s_primal_ctx_sqrt_0(float _S70)
{
    return (F32_sqrt((_S70)));
}

__device__ float3  s_primal_ctx_mul_1(float3  _S71, Matrix<float, 3, 3>  _S72)
{
    return mul_1(_S71, _S72);
}

__device__ void s_bwd_prop_lerp_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S73, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S74, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S75, float3  _S76)
{
    _d_lerp_vector_0(_S73, _S74, _S75, _S76);
    return;
}

__device__ void s_bwd_prop_lerp_1(DiffPair_float_0 * _S77, DiffPair_float_0 * _S78, DiffPair_float_0 * _S79, float _S80)
{
    _d_lerp_0(_S77, _S78, _S79, _S80);
    return;
}

__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S81, float _S82)
{
    _d_sqrt_0(_S81, _S82);
    return;
}

__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_7, float _s_dOut_0)
{
    float _S83 = (*dpx_7).primal_0.x;
    float _S84 = (*dpx_7).primal_0.y;
    float _S85 = (*dpx_7).primal_0.z;
    DiffPair_float_0 _S86;
    (&_S86)->primal_0 = _S83 * _S83 + _S84 * _S84 + _S85 * _S85;
    (&_S86)->differential_0 = 0.0f;
    s_bwd_prop_sqrt_0(&_S86, _s_dOut_0);
    float _S87 = (*dpx_7).primal_0.z * _S86.differential_0;
    float _S88 = _S87 + _S87;
    float _S89 = (*dpx_7).primal_0.y * _S86.differential_0;
    float _S90 = _S89 + _S89;
    float _S91 = (*dpx_7).primal_0.x * _S86.differential_0;
    float _S92 = _S91 + _S91;
    float3  _S93 = make_float3 (0.0f);
    *&((&_S93)->z) = _S88;
    *&((&_S93)->y) = _S90;
    *&((&_S93)->x) = _S92;
    dpx_7->primal_0 = (*dpx_7).primal_0;
    dpx_7->differential_0 = _S93;
    return;
}

__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S94, float _S95)
{
    s_bwd_prop_length_impl_0(_S94, _S95);
    return;
}

__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_8, float3  _s_dOut_1)
{
    float _S96 = length_0((*dpx_8).primal_0);
    float3  _S97 = (*dpx_8).primal_0 * _s_dOut_1;
    float3  _S98 = make_float3 (1.0f / _S96) * _s_dOut_1;
    float _S99 = - ((_S97.x + _S97.y + _S97.z) / (_S96 * _S96));
    float3  _S100 = make_float3 (0.0f);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S101;
    (&_S101)->primal_0 = (*dpx_8).primal_0;
    (&_S101)->differential_0 = _S100;
    s_bwd_length_impl_0(&_S101, _S99);
    float3  _S102 = _S98 + _S101.differential_0;
    dpx_8->primal_0 = (*dpx_8).primal_0;
    dpx_8->differential_0 = _S102;
    return;
}

__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S103, float3  _S104)
{
    s_bwd_prop_normalize_impl_0(_S103, _S104);
    return;
}

__device__ void s_bwd_prop_mul_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S105, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S106, float3  _S107)
{
    _d_mul_1(_S105, _S106, _S107);
    return;
}

__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S108, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S109, float _S110)
{
    _d_dot_0(_S108, _S109, _S110);
    return;
}

__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S111, DiffPair_float_0 * _S112, float _S113)
{
    _d_min_0(_S111, _S112, _S113);
    return;
}

__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S114, float _S115)
{
    _d_exp_0(_S114, _S115);
    return;
}

__device__ void s_bwd_prop_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S116, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S117, float3  _S118)
{
    _d_cross_0(_S116, _S117, _S118);
    return;
}

__device__ void s_bwd_prop_mul_1(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S119, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S120, float3  _S121)
{
    _d_mul_0(_S119, _S120, _S121);
    return;
}

struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_0;
    float4  differential_0;
};

__device__ void s_bwd_prop_transforms_rotationMatrixTranspose_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquaternion_1, Matrix<float, 3, 3>  _s_dOut_2)
{
    float _S122 = (*dpquaternion_1).primal_0.y;
    float _S123 = (*dpquaternion_1).primal_0.z;
    float _S124 = (*dpquaternion_1).primal_0.w;
    float _S125 = (*dpquaternion_1).primal_0.x;
    float _S126 = 2.0f * - _s_dOut_2.rows[int(2)].z;
    float _S127 = 2.0f * _s_dOut_2.rows[int(2)].y;
    float _S128 = 2.0f * _s_dOut_2.rows[int(2)].x;
    float _S129 = 2.0f * _s_dOut_2.rows[int(1)].z;
    float _S130 = 2.0f * - _s_dOut_2.rows[int(1)].y;
    float _S131 = 2.0f * _s_dOut_2.rows[int(1)].x;
    float _S132 = 2.0f * _s_dOut_2.rows[int(0)].z;
    float _S133 = 2.0f * _s_dOut_2.rows[int(0)].y;
    float _S134 = 2.0f * - _s_dOut_2.rows[int(0)].x;
    float _S135 = - _S131 + _S133;
    float _S136 = _S128 + - _S132;
    float _S137 = - _S127 + _S129;
    float _S138 = _S127 + _S129;
    float _S139 = _S128 + _S132;
    float _S140 = _S131 + _S133;
    float _S141 = _S124 * (_S130 + _S134);
    float _S142 = _S123 * (_S126 + _S134);
    float _S143 = _S122 * (_S126 + _S130);
    float4  _S144 = make_float4 (_S124 * _S135 + _S123 * _S136 + _S122 * _S137, _S125 * _S137 + _S124 * _S139 + _S123 * _S140 + _S143 + _S143, _S125 * _S136 + _S124 * _S138 + _S122 * _S140 + _S142 + _S142, _S125 * _S135 + _S123 * _S138 + _S122 * _S139 + _S141 + _S141);
    dpquaternion_1->primal_0 = (*dpquaternion_1).primal_0;
    dpquaternion_1->differential_0 = _S144;
    return;
}

__device__ void particleDensityProcessHitBwdToBuffer(float3  rayOrigin_2, float3  rayDirection_2, uint particleIdx_2, gaussianParticle_CommonParameters_0 commonParameters_2, float alpha_3, float alphaGrad_0, float * transmittance_2, float * transmittanceGrad_0, float depth_3, float * integratedDepth_2, float * integratedDepthGrad_0, bool enableNormal_3, float3  normal_3, float3  * integratedNormal_2, float3  * integratedNormalGrad_0)
{
    if(alpha_3 > 0.0f)
    {
        float weight_0 = 1.0f / (1.0f - alpha_3);
        float _S145 = *transmittance_2 * weight_0;
        *transmittance_2 = _S145;
        float _S146 = *transmittanceGrad_0;
        float _S147 = (*integratedDepth_2 - depth_3 * alpha_3) * weight_0;
        *integratedDepth_2 = _S147;
        float _S148 = *integratedDepthGrad_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 integratedNormalDiff_0;
        if(enableNormal_3)
        {
            float3  _S149 = (*integratedNormal_2 - normal_3 * make_float3 (alpha_3)) * make_float3 (weight_0);
            *integratedNormal_2 = _S149;
            (&integratedNormalDiff_0)->primal_0 = _S149;
            (&integratedNormalDiff_0)->differential_0 = *integratedNormalGrad_0;
        }
        else
        {
            float3  _S150 = make_float3 (0.0f);
            (&integratedNormalDiff_0)->primal_0 = _S150;
            (&integratedNormalDiff_0)->differential_0 = _S150;
        }
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S151 = integratedNormalDiff_0;
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S152;
        float3  _S153 = make_float3 (0.0f);
        float4  _S154 = make_float4 (0.0f);
        gaussianParticle_RawParameters_0 _S155 = { _S153, 0.0f, _S154, _S153, 0.0f };
        (&_S152)->_S53 = 0.0f;
        (&_S152)->_S54 = 0.0f;
        (&_S152)->_S55 = _S153;
        (&_S152)->_S56 = _S155;
        (&_S152)->_S53 = 0.0f;
        (&_S152)->_S54 = 0.0f;
        (&_S152)->_S55 = _S153;
        (&(&_S152)->_S56)->position_0 = _S153;
        (&(&_S152)->_S56)->density_0 = 0.0f;
        (&(&_S152)->_S56)->quaternion_0 = _S154;
        (&(&_S152)->_S56)->scale_0 = _S153;
        (&(&_S152)->_S56)->padding_0 = 0.0f;
        (&_S152)->_S53 = _S145;
        (&_S152)->_S54 = _S147;
        (&_S152)->_S55 = _S151.primal_0;
        (&_S152)->_S56 = *(commonParameters_2.parametersBuffer_0._dataPtr_0 + particleIdx_2);
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S156 = _S152;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S157 = integratedNormalDiff_0;
        Matrix<float, 3, 3>  _S158 = s_primal_ctx_transforms_rotationMatrixTranspose_0(_S152._S56.quaternion_0);
        float3  giscl_2 = make_float3 (1.0f) / _S152._S56.scale_0;
        float3  _S159 = _S152._S56.scale_0 * _S152._S56.scale_0;
        float3  gposc_0 = rayOrigin_2 - _S152._S56.position_0;
        float3  _S160 = s_primal_ctx_mul_0(_S158, gposc_0);
        float3  canonicalRayOrigin_2 = giscl_2 * _S160;
        float3  _S161 = s_primal_ctx_mul_0(_S158, rayDirection_2);
        float3  grdu_0 = giscl_2 * _S161;
        float3  _S162 = normalize_0(grdu_0);
        float3  _S163 = s_primal_ctx_cross_0(_S162, canonicalRayOrigin_2);
        float _S164 = s_primal_ctx_dot_0(_S163, _S163);
        float _S165 = -0.0555555559694767f * _S164;
        float _S166 = _S165 * _S164;
        float _S167 = s_primal_ctx_exp_0(_S166);
        float _S168 = _S167 * _S152._S56.density_0;
        float _S169 = s_primal_ctx_min_0(0.99000000953674316f, _S168);
        bool acceptHit_2;
        if(_S167 > 0.01130000036209822f)
        {
            acceptHit_2 = _S169 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_2 = false;
        }
        float3  normal_4;
        float3  _S170;
        float3  _S171;
        float3  _S172;
        float3  _S173;
        float3  _S174;
        float3  _S175;
        float3  _S176;
        float3  _S177;
        float depth_4;
        float _S178;
        if(acceptHit_2)
        {
            float3  _S179 = _S156._S56.scale_0 * _S162;
            float3  _S180 = make_float3 (-1.0f) * canonicalRayOrigin_2;
            float _S181 = s_primal_ctx_dot_0(_S162, _S180);
            float3  _S182 = make_float3 (_S181);
            float3  grds_2 = _S179 * make_float3 (_S181);
            float _S183 = s_primal_ctx_dot_0(grds_2, grds_2);
            float _S184 = s_primal_ctx_sqrt_0(_S183);
            if(enableNormal_3)
            {
                float3  surfelNm_4 = make_float3 (0.0f, 0.0f, 1.0f);
                if(s_primal_ctx_dot_0(surfelNm_4, _S162) > 0.0f)
                {
                    normal_4 = surfelNm_4 * make_float3 (-1.0f);
                }
                else
                {
                    normal_4 = surfelNm_4;
                }
                float3  _S185 = normal_4 * _S156._S56.scale_0;
                float3  _S186 = s_primal_ctx_mul_1(_S185, _S158);
                float3  _S187 = normalize_0(_S186);
                float3  _S188 = normal_4;
                normal_4 = _S187;
                _S170 = _S186;
                _S171 = _S185;
                _S172 = _S188;
                _S173 = surfelNm_4;
            }
            else
            {
                _S170 = _S153;
                _S171 = _S153;
                _S172 = _S153;
                _S173 = _S153;
            }
            depth_4 = _S184;
            _S178 = _S183;
            _S174 = grds_2;
            _S175 = _S179;
            _S176 = _S182;
            _S177 = _S180;
        }
        else
        {
            _S170 = _S153;
            _S171 = _S153;
            _S172 = _S153;
            _S173 = _S153;
            _S178 = 0.0f;
            _S174 = _S153;
            _S175 = _S153;
            _S176 = _S153;
            _S177 = _S153;
        }
        bool _bflag_0;
        float3  dpintegratedNormal_0;
        float _S189;
        if(acceptHit_2)
        {
            if(enableNormal_3)
            {
                dpintegratedNormal_0 = make_float3 (_S169);
            }
            else
            {
                dpintegratedNormal_0 = _S153;
            }
            float _S190 = 1.0f - _S169;
            _bflag_0 = false;
            _S189 = _S190;
        }
        else
        {
            _bflag_0 = true;
            _S189 = 0.0f;
            dpintegratedNormal_0 = _S153;
        }
        Matrix<float, 3, 3>  _S191 = makeMatrix<float, 3, 3> (0.0f);
        float dpintegratedDepth_0;
        if(_bflag_0)
        {
            dpintegratedDepth_0 = 0.0f;
        }
        else
        {
            dpintegratedDepth_0 = alphaGrad_0;
        }
        float dptransmittance_0;
        if(acceptHit_2)
        {
            float _S192 = _S189 * _S146;
            float _S193 = - (_S156._S53 * _S146) + dpintegratedDepth_0;
            if(enableNormal_3)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S194;
                (&_S194)->primal_0 = _S156._S55;
                (&_S194)->differential_0 = _S153;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S195;
                (&_S195)->primal_0 = normal_4;
                (&_S195)->differential_0 = _S153;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S196;
                (&_S196)->primal_0 = dpintegratedNormal_0;
                (&_S196)->differential_0 = _S153;
                s_bwd_prop_lerp_0(&_S194, &_S195, &_S196, _S157.differential_0);
                _S189 = _S196.differential_0.x + _S196.differential_0.y + _S196.differential_0.z + _S193;
                normal_4 = _S195.differential_0;
                dpintegratedNormal_0 = _S194.differential_0;
            }
            else
            {
                _S189 = _S193;
                normal_4 = _S153;
                dpintegratedNormal_0 = _S157.differential_0;
            }
            DiffPair_float_0 _S197;
            (&_S197)->primal_0 = _S156._S54;
            (&_S197)->differential_0 = 0.0f;
            DiffPair_float_0 _S198;
            (&_S198)->primal_0 = depth_4;
            (&_S198)->differential_0 = 0.0f;
            DiffPair_float_0 _S199;
            (&_S199)->primal_0 = _S169;
            (&_S199)->differential_0 = 0.0f;
            s_bwd_prop_lerp_1(&_S197, &_S198, &_S199, _S148);
            float _S200 = _S199.differential_0 + _S189;
            depth_4 = _S198.differential_0;
            _S189 = _S200;
            dpintegratedDepth_0 = _S197.differential_0;
            dptransmittance_0 = _S192;
        }
        else
        {
            depth_4 = 0.0f;
            normal_4 = _S153;
            _S189 = dpintegratedDepth_0;
            dpintegratedNormal_0 = _S157.differential_0;
            dpintegratedDepth_0 = _S148;
            dptransmittance_0 = _S146;
        }
        Matrix<float, 3, 3>  _S201;
        if(acceptHit_2)
        {
            if(enableNormal_3)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S202;
                (&_S202)->primal_0 = _S170;
                (&_S202)->differential_0 = _S153;
                s_bwd_normalize_impl_0(&_S202, normal_4);
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S203;
                (&_S203)->primal_0 = _S171;
                (&_S203)->differential_0 = _S153;
                DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S204;
                (&_S204)->primal_0 = _S158;
                (&_S204)->differential_0 = _S191;
                s_bwd_prop_mul_0(&_S203, &_S204, _S202.differential_0);
                float3  _S205 = _S172 * _S203.differential_0;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S206;
                (&_S206)->primal_0 = _S173;
                (&_S206)->differential_0 = _S153;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S207;
                (&_S207)->primal_0 = _S162;
                (&_S207)->differential_0 = _S153;
                s_bwd_prop_dot_0(&_S206, &_S207, 0.0f);
                normal_4 = _S207.differential_0;
                _S170 = _S205;
                _S201 = _S204.differential_0;
            }
            else
            {
                normal_4 = _S153;
                _S170 = _S153;
                _S201 = _S191;
            }
            DiffPair_float_0 _S208;
            (&_S208)->primal_0 = _S178;
            (&_S208)->differential_0 = 0.0f;
            s_bwd_prop_sqrt_0(&_S208, depth_4);
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S209;
            (&_S209)->primal_0 = _S174;
            (&_S209)->differential_0 = _S153;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S210;
            (&_S210)->primal_0 = _S174;
            (&_S210)->differential_0 = _S153;
            s_bwd_prop_dot_0(&_S209, &_S210, _S208.differential_0);
            float3  _S211 = _S210.differential_0 + _S209.differential_0;
            float3  _S212 = _S175 * _S211;
            float3  _S213 = _S176 * _S211;
            float _S214 = _S212.x + _S212.y + _S212.z;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S215;
            (&_S215)->primal_0 = _S162;
            (&_S215)->differential_0 = _S153;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S216;
            (&_S216)->primal_0 = _S177;
            (&_S216)->differential_0 = _S153;
            s_bwd_prop_dot_0(&_S215, &_S216, _S214);
            float3  _S217 = make_float3 (-1.0f) * _S216.differential_0;
            float3  _S218 = _S162 * _S213 + _S170;
            normal_4 = _S215.differential_0 + _S156._S56.scale_0 * _S213 + normal_4;
            _S170 = _S217;
            _S171 = _S218;
        }
        else
        {
            normal_4 = _S153;
            _S170 = _S153;
            _S201 = _S191;
            _S171 = _S153;
        }
        DiffPair_float_0 _S219;
        (&_S219)->primal_0 = 0.99000000953674316f;
        (&_S219)->differential_0 = 0.0f;
        DiffPair_float_0 _S220;
        (&_S220)->primal_0 = _S168;
        (&_S220)->differential_0 = 0.0f;
        s_bwd_prop_min_0(&_S219, &_S220, _S189);
        float _S221 = _S167 * _S220.differential_0;
        float _S222 = _S156._S56.density_0 * _S220.differential_0;
        DiffPair_float_0 _S223;
        (&_S223)->primal_0 = _S166;
        (&_S223)->differential_0 = 0.0f;
        s_bwd_prop_exp_0(&_S223, _S222);
        float _S224 = _S165 * _S223.differential_0 + -0.0555555559694767f * (_S164 * _S223.differential_0);
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S225;
        (&_S225)->primal_0 = _S163;
        (&_S225)->differential_0 = _S153;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S226;
        (&_S226)->primal_0 = _S163;
        (&_S226)->differential_0 = _S153;
        s_bwd_prop_dot_0(&_S225, &_S226, _S224);
        float3  _S227 = _S226.differential_0 + _S225.differential_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S228;
        (&_S228)->primal_0 = _S162;
        (&_S228)->differential_0 = _S153;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S229;
        (&_S229)->primal_0 = canonicalRayOrigin_2;
        (&_S229)->differential_0 = _S153;
        s_bwd_prop_cross_0(&_S228, &_S229, _S227);
        float3  _S230 = _S228.differential_0 + normal_4;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S231;
        (&_S231)->primal_0 = grdu_0;
        (&_S231)->differential_0 = _S153;
        s_bwd_normalize_impl_0(&_S231, _S230);
        float3  _S232 = giscl_2 * _S231.differential_0;
        float3  _S233 = _S161 * _S231.differential_0;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S234;
        (&_S234)->primal_0 = _S158;
        (&_S234)->differential_0 = _S191;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S235;
        (&_S235)->primal_0 = rayDirection_2;
        (&_S235)->differential_0 = _S153;
        s_bwd_prop_mul_1(&_S234, &_S235, _S232);
        float3  _S236 = _S229.differential_0 + _S170;
        float3  _S237 = giscl_2 * _S236;
        float3  _S238 = _S160 * _S236;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S239;
        (&_S239)->primal_0 = _S158;
        (&_S239)->differential_0 = _S191;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S240;
        (&_S240)->primal_0 = gposc_0;
        (&_S240)->differential_0 = _S153;
        s_bwd_prop_mul_1(&_S239, &_S240, _S237);
        float3  _S241 = - _S240.differential_0;
        float3  _S242 = - ((_S233 + _S238) / _S159);
        Matrix<float, 3, 3>  _S243 = _S234.differential_0 + _S239.differential_0 + _S201;
        DiffPair_vectorx3Cfloatx2C4x3E_0 _S244;
        (&_S244)->primal_0 = _S156._S56.quaternion_0;
        (&_S244)->differential_0 = _S154;
        s_bwd_prop_transforms_rotationMatrixTranspose_0(&_S244, _S243);
        float3  _S245 = _S242 + _S171;
        gaussianParticle_RawParameters_0 _S246 = gaussianParticle_RawParameters_x24_syn_dzero_0();
        (&_S246)->density_0 = _S221;
        (&_S246)->quaternion_0 = _S244.differential_0;
        (&_S246)->scale_0 = _S245;
        (&_S246)->position_0 = _S241;
        gaussianParticle_RawParameters_0 _S247 = _S246;
        if(commonParameters_2.parametersBuffer_0.exclusiveGradient_0)
        {
            gaussianParticle_RawParameters_0 * _S248 = commonParameters_2.parametersBuffer_0._gradPtr_0 + particleIdx_2;
            _S248->density_0 = _S248->density_0 + _S247.density_0;
            _S248->position_0 = _S248->position_0 + _S247.position_0;
            _S248->quaternion_0 = _S248->quaternion_0 + _S247.quaternion_0;
            _S248->scale_0 = _S248->scale_0 + _S247.scale_0;
        }
        else
        {
            gaussianParticle_RawParameters_0 * _S249 = commonParameters_2.parametersBuffer_0._gradPtr_0 + particleIdx_2;
            float _S250 = atomicAdd(&_S249->density_0, _S247.density_0);
            float _S251 = atomicAdd(&((&_S249->position_0)->x), _S247.position_0.x);
            float _S252 = atomicAdd(&((&_S249->position_0)->y), _S247.position_0.y);
            float _S253 = atomicAdd(&((&_S249->position_0)->z), _S247.position_0.z);
            float _S254 = atomicAdd(&((&_S249->quaternion_0)->x), _S247.quaternion_0.x);
            float _S255 = atomicAdd(&((&_S249->quaternion_0)->y), _S247.quaternion_0.y);
            float _S256 = atomicAdd(&((&_S249->quaternion_0)->z), _S247.quaternion_0.z);
            float _S257 = atomicAdd(&((&_S249->quaternion_0)->w), _S247.quaternion_0.w);
            float _S258 = atomicAdd(&((&_S249->scale_0)->x), _S247.scale_0.x);
            float _S259 = atomicAdd(&((&_S249->scale_0)->y), _S247.scale_0.y);
            float _S260 = atomicAdd(&((&_S249->scale_0)->z), _S247.scale_0.z);
        }
        *transmittanceGrad_0 = dptransmittance_0;
        *integratedDepthGrad_0 = dpintegratedDepth_0;
        if(enableNormal_3)
        {
            *integratedNormalGrad_0 = dpintegratedNormal_0;
        }
    }
    return;
}

__device__ bool particleDensityHitCustom(float3  rayOrigin_3, float3  rayDirection_3, int particleIdx_3, gaussianParticle_CommonParameters_0 commonParameters_3, float minHitDistance_0, float maxHitDistance_0, float maxParticleSquaredDistance_0, float * hitDistance_0)
{
    gaussianParticle_RawParameters_0 * _S261 = commonParameters_3.parametersBuffer_0._dataPtr_0 + uint(particleIdx_3);
    Matrix<float, 3, 3>  _S262 = transforms_rotationMatrixTranspose_0((*_S261).quaternion_0);
    float3  giscl_3 = make_float3 (1.0f) / (*_S261).scale_0;
    float3  canonicalRayOrigin_3 = giscl_3 * mul_0(_S262, rayOrigin_3 - (*_S261).position_0);
    float3  canonicalRayDirection_2 = normalize_0(giscl_3 * mul_0(_S262, rayDirection_3));
    float3  grds_3 = (*_S261).scale_0 * canonicalRayDirection_2 * make_float3 (dot_0(canonicalRayDirection_2, make_float3 (-1.0f) * canonicalRayOrigin_3));
    float _S263 = (F32_sqrt((dot_0(grds_3, grds_3))));
    *hitDistance_0 = _S263;
    bool _S264;
    if(_S263 > minHitDistance_0)
    {
        _S264 = *hitDistance_0 < maxHitDistance_0;
    }
    else
    {
        _S264 = false;
    }
    if(_S264)
    {
        float3  gcrod_2 = cross_0(canonicalRayDirection_2, canonicalRayOrigin_3);
        _S264 = dot_0(gcrod_2, gcrod_2) < maxParticleSquaredDistance_0;
    }
    else
    {
        _S264 = false;
    }
    return _S264;
}

__device__ float rcp_0(float x_3)
{
    return 1.0f / x_3;
}

__device__ bool particleDensityHitInstance(float3  canonicalRayOrigin_4, float3  canonicalUnormalizedRayDirection_0, float minHitDistance_1, float maxHitDistance_1, float maxParticleSquaredDistance_1, float * hitDistance_1)
{
    float _S265 = - dot_0(canonicalRayOrigin_4, canonicalUnormalizedRayDirection_0) * rcp_0(dot_0(canonicalUnormalizedRayDirection_0, canonicalUnormalizedRayDirection_0));
    *hitDistance_1 = _S265;
    bool _S266;
    if(_S265 > minHitDistance_1)
    {
        _S266 = *hitDistance_1 < maxHitDistance_1;
    }
    else
    {
        _S266 = false;
    }
    if(_S266)
    {
        float3  gcrod_3 = cross_0(normalize_0(canonicalUnormalizedRayDirection_0), canonicalRayOrigin_4);
        _S266 = dot_0(gcrod_3, gcrod_3) < maxParticleSquaredDistance_1;
    }
    else
    {
        _S266 = false;
    }
    return _S266;
}

__device__ float3  particleDensityIncidentDirection(gaussianParticle_Parameters_0 parameters_1, float3  sourcePosition_0)
{
    return normalize_0(parameters_1.position_1 - sourcePosition_0);
}

struct s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0
{
    gaussianParticle_RawParameters_0 _S267;
};

__device__ void particleDensityIncidentDirectionBwdToBuffer(uint particleIdx_4, gaussianParticle_CommonParameters_0 commonParameters_4, float3  sourcePosition_1, float3  incidentDirectionGrad_0)
{
    float3  _S268 = make_float3 (0.0f);
    float4  _S269 = make_float4 (0.0f);
    gaussianParticle_RawParameters_0 _S270 = { _S268, 0.0f, _S269, _S268, 0.0f };
    s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0 _S271;
    (&_S271)->_S267 = _S270;
    (&(&_S271)->_S267)->position_0 = _S268;
    (&(&_S271)->_S267)->density_0 = 0.0f;
    (&(&_S271)->_S267)->quaternion_0 = _S269;
    (&(&_S271)->_S267)->scale_0 = _S268;
    (&(&_S271)->_S267)->padding_0 = 0.0f;
    (&_S271)->_S267 = *(commonParameters_4.parametersBuffer_0._dataPtr_0 + particleIdx_4);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S272;
    (&_S272)->primal_0 = _S271._S267.position_0 - sourcePosition_1;
    (&_S272)->differential_0 = _S268;
    s_bwd_normalize_impl_0(&_S272, incidentDirectionGrad_0);
    gaussianParticle_RawParameters_0 _S273 = gaussianParticle_RawParameters_x24_syn_dzero_0();
    (&_S273)->position_0 = _S272.differential_0;
    gaussianParticle_RawParameters_0 _S274 = _S273;
    if(commonParameters_4.parametersBuffer_0.exclusiveGradient_0)
    {
        gaussianParticle_RawParameters_0 * _S275 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
        _S275->density_0 = _S275->density_0 + _S274.density_0;
        _S275->position_0 = _S275->position_0 + _S274.position_0;
        _S275->quaternion_0 = _S275->quaternion_0 + _S274.quaternion_0;
        _S275->scale_0 = _S275->scale_0 + _S274.scale_0;
    }
    else
    {
        gaussianParticle_RawParameters_0 * _S276 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
        float _S277 = atomicAdd(&_S276->density_0, _S274.density_0);
        float _S278 = atomicAdd(&((&_S276->position_0)->x), _S274.position_0.x);
        float _S279 = atomicAdd(&((&_S276->position_0)->y), _S274.position_0.y);
        float _S280 = atomicAdd(&((&_S276->position_0)->z), _S274.position_0.z);
        float _S281 = atomicAdd(&((&_S276->quaternion_0)->x), _S274.quaternion_0.x);
        float _S282 = atomicAdd(&((&_S276->quaternion_0)->y), _S274.quaternion_0.y);
        float _S283 = atomicAdd(&((&_S276->quaternion_0)->z), _S274.quaternion_0.z);
        float _S284 = atomicAdd(&((&_S276->quaternion_0)->w), _S274.quaternion_0.w);
        float _S285 = atomicAdd(&((&_S276->scale_0)->x), _S274.scale_0.x);
        float _S286 = atomicAdd(&((&_S276->scale_0)->y), _S274.scale_0.y);
        float _S287 = atomicAdd(&((&_S276->scale_0)->z), _S274.scale_0.z);
    }
    return;
}

struct shRadiativeParticle_Parameters_0
{
    FixedArray<float3 , 16>  sphCoefficients_0;
};

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dzero_0()
{
    shRadiativeParticle_Parameters_0 result_5;
    float3  _S288 = make_float3 (0.0f);
    (&result_5)->sphCoefficients_0[int(0)] = _S288;
    (&result_5)->sphCoefficients_0[int(1)] = _S288;
    (&result_5)->sphCoefficients_0[int(2)] = _S288;
    (&result_5)->sphCoefficients_0[int(3)] = _S288;
    (&result_5)->sphCoefficients_0[int(4)] = _S288;
    (&result_5)->sphCoefficients_0[int(5)] = _S288;
    (&result_5)->sphCoefficients_0[int(6)] = _S288;
    (&result_5)->sphCoefficients_0[int(7)] = _S288;
    (&result_5)->sphCoefficients_0[int(8)] = _S288;
    (&result_5)->sphCoefficients_0[int(9)] = _S288;
    (&result_5)->sphCoefficients_0[int(10)] = _S288;
    (&result_5)->sphCoefficients_0[int(11)] = _S288;
    (&result_5)->sphCoefficients_0[int(12)] = _S288;
    (&result_5)->sphCoefficients_0[int(13)] = _S288;
    (&result_5)->sphCoefficients_0[int(14)] = _S288;
    (&result_5)->sphCoefficients_0[int(15)] = _S288;
    return result_5;
}

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dadd_0(shRadiativeParticle_Parameters_0 SLANG_anonymous_0_0, shRadiativeParticle_Parameters_0 SLANG_anonymous_1_0)
{
    shRadiativeParticle_Parameters_0 result_6;
    (&result_6)->sphCoefficients_0[int(0)] = SLANG_anonymous_0_0.sphCoefficients_0[int(0)] + SLANG_anonymous_1_0.sphCoefficients_0[int(0)];
    (&result_6)->sphCoefficients_0[int(1)] = SLANG_anonymous_0_0.sphCoefficients_0[int(1)] + SLANG_anonymous_1_0.sphCoefficients_0[int(1)];
    (&result_6)->sphCoefficients_0[int(2)] = SLANG_anonymous_0_0.sphCoefficients_0[int(2)] + SLANG_anonymous_1_0.sphCoefficients_0[int(2)];
    (&result_6)->sphCoefficients_0[int(3)] = SLANG_anonymous_0_0.sphCoefficients_0[int(3)] + SLANG_anonymous_1_0.sphCoefficients_0[int(3)];
    (&result_6)->sphCoefficients_0[int(4)] = SLANG_anonymous_0_0.sphCoefficients_0[int(4)] + SLANG_anonymous_1_0.sphCoefficients_0[int(4)];
    (&result_6)->sphCoefficients_0[int(5)] = SLANG_anonymous_0_0.sphCoefficients_0[int(5)] + SLANG_anonymous_1_0.sphCoefficients_0[int(5)];
    (&result_6)->sphCoefficients_0[int(6)] = SLANG_anonymous_0_0.sphCoefficients_0[int(6)] + SLANG_anonymous_1_0.sphCoefficients_0[int(6)];
    (&result_6)->sphCoefficients_0[int(7)] = SLANG_anonymous_0_0.sphCoefficients_0[int(7)] + SLANG_anonymous_1_0.sphCoefficients_0[int(7)];
    (&result_6)->sphCoefficients_0[int(8)] = SLANG_anonymous_0_0.sphCoefficients_0[int(8)] + SLANG_anonymous_1_0.sphCoefficients_0[int(8)];
    (&result_6)->sphCoefficients_0[int(9)] = SLANG_anonymous_0_0.sphCoefficients_0[int(9)] + SLANG_anonymous_1_0.sphCoefficients_0[int(9)];
    (&result_6)->sphCoefficients_0[int(10)] = SLANG_anonymous_0_0.sphCoefficients_0[int(10)] + SLANG_anonymous_1_0.sphCoefficients_0[int(10)];
    (&result_6)->sphCoefficients_0[int(11)] = SLANG_anonymous_0_0.sphCoefficients_0[int(11)] + SLANG_anonymous_1_0.sphCoefficients_0[int(11)];
    (&result_6)->sphCoefficients_0[int(12)] = SLANG_anonymous_0_0.sphCoefficients_0[int(12)] + SLANG_anonymous_1_0.sphCoefficients_0[int(12)];
    (&result_6)->sphCoefficients_0[int(13)] = SLANG_anonymous_0_0.sphCoefficients_0[int(13)] + SLANG_anonymous_1_0.sphCoefficients_0[int(13)];
    (&result_6)->sphCoefficients_0[int(14)] = SLANG_anonymous_0_0.sphCoefficients_0[int(14)] + SLANG_anonymous_1_0.sphCoefficients_0[int(14)];
    (&result_6)->sphCoefficients_0[int(15)] = SLANG_anonymous_0_0.sphCoefficients_0[int(15)] + SLANG_anonymous_1_0.sphCoefficients_0[int(15)];
    return result_6;
}

__device__ void _d_max_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_9, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_5, float3  dOut_10)
{
    DiffPair_float_0 left_dp_3;
    (&left_dp_3)->primal_0 = (*dpx_9).primal_0.x;
    (&left_dp_3)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_3;
    (&right_dp_3)->primal_0 = (*dpy_5).primal_0.x;
    (&right_dp_3)->differential_0 = 0.0f;
    _d_max_0(&left_dp_3, &right_dp_3, dOut_10.x);
    float3  left_d_result_3;
    *&((&left_d_result_3)->x) = left_dp_3.differential_0;
    float3  right_d_result_3;
    *&((&right_d_result_3)->x) = right_dp_3.differential_0;
    DiffPair_float_0 left_dp_4;
    (&left_dp_4)->primal_0 = (*dpx_9).primal_0.y;
    (&left_dp_4)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_4;
    (&right_dp_4)->primal_0 = (*dpy_5).primal_0.y;
    (&right_dp_4)->differential_0 = 0.0f;
    _d_max_0(&left_dp_4, &right_dp_4, dOut_10.y);
    *&((&left_d_result_3)->y) = left_dp_4.differential_0;
    *&((&right_d_result_3)->y) = right_dp_4.differential_0;
    DiffPair_float_0 left_dp_5;
    (&left_dp_5)->primal_0 = (*dpx_9).primal_0.z;
    (&left_dp_5)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_5;
    (&right_dp_5)->primal_0 = (*dpy_5).primal_0.z;
    (&right_dp_5)->differential_0 = 0.0f;
    _d_max_0(&left_dp_5, &right_dp_5, dOut_10.z);
    *&((&left_d_result_3)->z) = left_dp_5.differential_0;
    *&((&right_d_result_3)->z) = right_dp_5.differential_0;
    dpx_9->primal_0 = (*dpx_9).primal_0;
    dpx_9->differential_0 = left_d_result_3;
    dpy_5->primal_0 = (*dpy_5).primal_0;
    dpy_5->differential_0 = right_d_result_3;
    return;
}

__device__ float3  max_0(float3  x_4, float3  y_1)
{
    float3  result_7;
    int i_3 = int(0);
    for(;;)
    {
        if(i_3 < int(3))
        {
        }
        else
        {
            break;
        }
        *_slang_vector_get_element_ptr(&result_7, i_3) = (F32_max((_slang_vector_get_element(x_4, i_3)), (_slang_vector_get_element(y_1, i_3))));
        i_3 = i_3 + int(1);
    }
    return result_7;
}

__device__ float3  sphericalHarmonics_decode_0(int degree_0, FixedArray<float3 , 16>  coefficients_0, float3  direction_0)
{
    float3  features_0 = make_float3 (0.282094806432724f) * coefficients_0[int(0)];
    float3  features_1;
    if(degree_0 > int(0))
    {
        float x_5 = direction_0.x;
        float y_2 = direction_0.y;
        float z_0 = direction_0.z;
        float3  features_2 = features_0 - make_float3 (0.48860251903533936f * y_2) * coefficients_0[int(1)] + make_float3 (0.48860251903533936f * z_0) * coefficients_0[int(2)] - make_float3 (0.48860251903533936f * x_5) * coefficients_0[int(3)];
        if(degree_0 > int(1))
        {
            float xx_2 = x_5 * x_5;
            float yy_2 = y_2 * y_2;
            float zz_2 = z_0 * z_0;
            float xy_2 = x_5 * y_2;
            float _S289 = 2.0f * zz_2;
            float _S290 = xx_2 - yy_2;
            float3  features_3 = features_2 + make_float3 (1.09254848957061768f * xy_2) * coefficients_0[int(4)] + make_float3 (-1.09254848957061768f * (y_2 * z_0)) * coefficients_0[int(5)] + make_float3 (0.31539157032966614f * (_S289 - xx_2 - yy_2)) * coefficients_0[int(6)] + make_float3 (-1.09254848957061768f * (x_5 * z_0)) * coefficients_0[int(7)] + make_float3 (0.54627424478530884f * _S290) * coefficients_0[int(8)];
            if(degree_0 > int(2))
            {
                float _S291 = 3.0f * xx_2;
                float _S292 = 4.0f * zz_2 - xx_2 - yy_2;
                float _S293 = 3.0f * yy_2;
                features_1 = features_3 + make_float3 (-0.59004360437393188f * y_2 * (_S291 - yy_2)) * coefficients_0[int(9)] + make_float3 (2.89061141014099121f * xy_2 * z_0) * coefficients_0[int(10)] + make_float3 (-0.4570457935333252f * y_2 * _S292) * coefficients_0[int(11)] + make_float3 (0.37317633628845215f * z_0 * (_S289 - _S291 - _S293)) * coefficients_0[int(12)] + make_float3 (-0.4570457935333252f * x_5 * _S292) * coefficients_0[int(13)] + make_float3 (1.44530570507049561f * z_0 * _S290) * coefficients_0[int(14)] + make_float3 (-0.59004360437393188f * x_5 * (xx_2 - _S293)) * coefficients_0[int(15)];
            }
            else
            {
                features_1 = features_3;
            }
        }
        else
        {
            features_1 = features_2;
        }
    }
    else
    {
        features_1 = features_0;
    }
    return max_0(features_1 + make_float3 (0.5f), make_float3 (0.0f));
}

struct shRadiativeParticle_ParametersBuffer_0
{
    float3  * _dataPtr_1;
    float3  * _gradPtr_1;
    bool exclusiveGradient_1;
};

struct shRadiativeParticle_CommonParameters_0
{
    shRadiativeParticle_ParametersBuffer_0 parametersBuffer_1;
    int sphDegree_0;
};

__device__ float3  particleFeaturesFromBuffer(uint particleIdx_5, shRadiativeParticle_CommonParameters_0 commonParameters_5, float3  incidentDirection_0)
{
    shRadiativeParticle_Parameters_0 _S294;
    uint _S295 = particleIdx_5 * 16U;
    for(;;)
    {
        int i_4 = int(0);
        shRadiativeParticle_Parameters_0 parameters_2;
        #pragma unroll
        for(;;)
        {
            if(i_4 < int(16))
            {
            }
            else
            {
                break;
            }
            (&parameters_2)->sphCoefficients_0[i_4] = *(commonParameters_5.parametersBuffer_1._dataPtr_1 + (_S295 + uint(i_4)));
            i_4 = i_4 + int(1);
        }
        _S294 = parameters_2;
        break;
    }
    return sphericalHarmonics_decode_0(commonParameters_5.sphDegree_0, _S294.sphCoefficients_0, incidentDirection_0);
}

__device__ void particleFeaturesIntegrateFwd(float weight_1, float3  features_4, float3  * integratedFeatures_0)
{
    if(weight_1 > 0.0f)
    {
        *integratedFeatures_0 = *integratedFeatures_0 + features_4 * make_float3 (weight_1);
    }
    return;
}

__device__ void particleFeaturesIntegrateFwdFromBuffer(float3  incidentDirection_1, float weight_2, uint particleIdx_6, shRadiativeParticle_CommonParameters_0 commonParameters_6, float3  * integratedFeatures_1)
{
    shRadiativeParticle_Parameters_0 _S296;
    uint _S297 = particleIdx_6 * 16U;
    bool _S298 = weight_2 > 0.0f;
    int _S299 = int(uint(commonParameters_6.sphDegree_0));
    for(;;)
    {
        for(;;)
        {
            int i_5 = int(0);
            shRadiativeParticle_Parameters_0 parameters_3;
            #pragma unroll
            for(;;)
            {
                if(i_5 < int(16))
                {
                }
                else
                {
                    break;
                }
                (&parameters_3)->sphCoefficients_0[i_5] = *(commonParameters_6.parametersBuffer_1._dataPtr_1 + (_S297 + uint(i_5)));
                i_5 = i_5 + int(1);
            }
            _S296 = parameters_3;
            break;
        }
        if(_S298)
        {
            float3  _S300 = sphericalHarmonics_decode_0(_S299, _S296.sphCoefficients_0, incidentDirection_1);
            if(_S298)
            {
                *integratedFeatures_1 = *integratedFeatures_1 + _S300 * make_float3 (weight_2);
            }
        }
        break;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0
{
    float3  _S301;
};

__device__ void particleFeaturesIntegrateBwd(float alpha_4, float * alphaGrad_1, float3  features_5, float3  * featuresGrad_0, float3  * integratedFeatures_2, float3  * integratedFeaturesGrad_0)
{
    bool _S302 = alpha_4 > 0.0f;
    if(_S302)
    {
        float3  _S303 = (*integratedFeatures_2 - features_5 * make_float3 (alpha_4)) * make_float3 (1.0f / (1.0f - alpha_4));
        *integratedFeatures_2 = _S303;
        float3  _S304 = *integratedFeaturesGrad_0;
        float3  _S305 = make_float3 (0.0f);
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S306;
        (&_S306)->_S301 = _S305;
        (&_S306)->_S301 = _S303;
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S307 = _S306;
        float3  dpintegratedRadiance_0;
        if(_S302)
        {
            dpintegratedRadiance_0 = make_float3 (alpha_4);
        }
        else
        {
            dpintegratedRadiance_0 = _S305;
        }
        float3  _S308;
        float _S309;
        if(_S302)
        {
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S310;
            (&_S310)->primal_0 = _S307._S301;
            (&_S310)->differential_0 = _S305;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S311;
            (&_S311)->primal_0 = features_5;
            (&_S311)->differential_0 = _S305;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S312;
            (&_S312)->primal_0 = dpintegratedRadiance_0;
            (&_S312)->differential_0 = _S305;
            s_bwd_prop_lerp_0(&_S310, &_S311, &_S312, _S304);
            float _S313 = _S312.differential_0.x + _S312.differential_0.y + _S312.differential_0.z;
            dpintegratedRadiance_0 = _S310.differential_0;
            _S308 = _S311.differential_0;
            _S309 = _S313;
        }
        else
        {
            dpintegratedRadiance_0 = _S304;
            _S308 = _S305;
            _S309 = 0.0f;
        }
        *alphaGrad_1 = _S309;
        *featuresGrad_0 = _S308;
        *integratedFeaturesGrad_0 = dpintegratedRadiance_0;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0
{
    float3  _S314;
    shRadiativeParticle_Parameters_0 _S315;
};

__device__ float3  s_primal_ctx_max_0(float3  _S316, float3  _S317)
{
    return max_0(_S316, _S317);
}

__device__ float3  s_primal_ctx_sphericalHarmonics_decode_0(int degree_1, FixedArray<float3 , 16>  dpcoefficients_0, float3  dpdirection_0)
{
    float3  features_6 = make_float3 (0.282094806432724f) * dpcoefficients_0[int(0)];
    float3  features_7;
    if(degree_1 > int(0))
    {
        float x_6 = dpdirection_0.x;
        float y_3 = dpdirection_0.y;
        float z_1 = dpdirection_0.z;
        float3  features_8 = features_6 - make_float3 (0.48860251903533936f * y_3) * dpcoefficients_0[int(1)] + make_float3 (0.48860251903533936f * z_1) * dpcoefficients_0[int(2)] - make_float3 (0.48860251903533936f * x_6) * dpcoefficients_0[int(3)];
        if(degree_1 > int(1))
        {
            float xx_3 = x_6 * x_6;
            float yy_3 = y_3 * y_3;
            float zz_3 = z_1 * z_1;
            float xy_3 = x_6 * y_3;
            float _S318 = 2.0f * zz_3;
            float _S319 = xx_3 - yy_3;
            float3  features_9 = features_8 + make_float3 (1.09254848957061768f * xy_3) * dpcoefficients_0[int(4)] + make_float3 (-1.09254848957061768f * (y_3 * z_1)) * dpcoefficients_0[int(5)] + make_float3 (0.31539157032966614f * (_S318 - xx_3 - yy_3)) * dpcoefficients_0[int(6)] + make_float3 (-1.09254848957061768f * (x_6 * z_1)) * dpcoefficients_0[int(7)] + make_float3 (0.54627424478530884f * _S319) * dpcoefficients_0[int(8)];
            if(degree_1 > int(2))
            {
                float _S320 = 3.0f * xx_3;
                float _S321 = 4.0f * zz_3 - xx_3 - yy_3;
                float _S322 = 3.0f * yy_3;
                features_7 = features_9 + make_float3 (-0.59004360437393188f * y_3 * (_S320 - yy_3)) * dpcoefficients_0[int(9)] + make_float3 (2.89061141014099121f * xy_3 * z_1) * dpcoefficients_0[int(10)] + make_float3 (-0.4570457935333252f * y_3 * _S321) * dpcoefficients_0[int(11)] + make_float3 (0.37317633628845215f * z_1 * (_S318 - _S320 - _S322)) * dpcoefficients_0[int(12)] + make_float3 (-0.4570457935333252f * x_6 * _S321) * dpcoefficients_0[int(13)] + make_float3 (1.44530570507049561f * z_1 * _S319) * dpcoefficients_0[int(14)] + make_float3 (-0.59004360437393188f * x_6 * (xx_3 - _S322)) * dpcoefficients_0[int(15)];
            }
            else
            {
                features_7 = features_9;
            }
        }
        else
        {
            features_7 = features_8;
        }
    }
    else
    {
        features_7 = features_6;
    }
    return s_primal_ctx_max_0(features_7 + make_float3 (0.5f), make_float3 (0.0f));
}

struct DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0
{
    FixedArray<float3 , 16>  primal_0;
    FixedArray<float3 , 16>  differential_0;
};

__device__ void s_bwd_prop_max_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S323, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S324, float3  _S325)
{
    _d_max_vector_0(_S323, _S324, _S325);
    return;
}

__device__ void s_bwd_prop_sphericalHarmonics_decode_0(int degree_2, DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 * dpcoefficients_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdirection_1, float3  _s_dOut_3)
{
    DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S326 = *dpcoefficients_1;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S327 = *dpdirection_1;
    float3  _S328 = make_float3 (0.0f);
    float3  features_10 = make_float3 (0.282094806432724f) * (*dpcoefficients_1).primal_0[int(0)];
    bool _S329 = degree_2 > int(0);
    float3  features_11;
    float3  _S330;
    float3  _S331;
    float3  _S332;
    float3  _S333;
    float3  _S334;
    float3  _S335;
    float3  _S336;
    float3  _S337;
    float3  _S338;
    float3  _S339;
    float3  _S340;
    float3  _S341;
    float3  _S342;
    float3  _S343;
    float3  _S344;
    float3  _S345;
    float3  _S346;
    float3  _S347;
    float3  _S348;
    float3  _S349;
    float3  _S350;
    float3  _S351;
    float3  _S352;
    float3  _S353;
    float3  _S354;
    float3  _S355;
    float3  _S356;
    float3  _S357;
    float3  _S358;
    float3  _S359;
    float _S360;
    float _S361;
    float _S362;
    float _S363;
    float _S364;
    float _S365;
    float _S366;
    float _S367;
    float _S368;
    float _S369;
    float _S370;
    float _S371;
    float _S372;
    float _S373;
    float _S374;
    bool _S375;
    bool _S376;
    if(_S329)
    {
        float x_7 = _S327.primal_0.x;
        float y_4 = _S327.primal_0.y;
        float z_2 = _S327.primal_0.z;
        float _S377 = 0.48860251903533936f * y_4;
        float3  _S378 = make_float3 (_S377);
        float _S379 = 0.48860251903533936f * z_2;
        float3  _S380 = make_float3 (_S379);
        float _S381 = 0.48860251903533936f * x_7;
        float3  _S382 = make_float3 (_S381);
        float3  features_12 = features_10 - make_float3 (_S377) * _S326.primal_0[int(1)] + make_float3 (_S379) * _S326.primal_0[int(2)] - make_float3 (_S381) * _S326.primal_0[int(3)];
        bool _S383 = degree_2 > int(1);
        if(_S383)
        {
            float xx_4 = x_7 * x_7;
            float yy_4 = y_4 * y_4;
            float zz_4 = z_2 * z_2;
            float xy_4 = x_7 * y_4;
            float _S384 = 1.09254848957061768f * xy_4;
            float3  _S385 = make_float3 (_S384);
            float _S386 = -1.09254848957061768f * (y_4 * z_2);
            float3  _S387 = make_float3 (_S386);
            float _S388 = 2.0f * zz_4;
            float _S389 = 0.31539157032966614f * (_S388 - xx_4 - yy_4);
            float3  _S390 = make_float3 (_S389);
            float _S391 = -1.09254848957061768f * (x_7 * z_2);
            float3  _S392 = make_float3 (_S391);
            float _S393 = xx_4 - yy_4;
            float _S394 = 0.54627424478530884f * _S393;
            float3  _S395 = make_float3 (_S394);
            float3  features_13 = features_12 + make_float3 (_S384) * _S326.primal_0[int(4)] + make_float3 (_S386) * _S326.primal_0[int(5)] + make_float3 (_S389) * _S326.primal_0[int(6)] + make_float3 (_S391) * _S326.primal_0[int(7)] + make_float3 (_S394) * _S326.primal_0[int(8)];
            bool _S396 = degree_2 > int(2);
            if(_S396)
            {
                float _S397 = -0.59004360437393188f * y_4;
                float _S398 = 3.0f * xx_4;
                float _S399 = _S398 - yy_4;
                float _S400 = _S397 * _S399;
                float3  _S401 = make_float3 (_S400);
                float _S402 = 2.89061141014099121f * xy_4;
                float _S403 = _S402 * z_2;
                float3  _S404 = make_float3 (_S403);
                float _S405 = -0.4570457935333252f * y_4;
                float _S406 = 4.0f * zz_4 - xx_4 - yy_4;
                float _S407 = _S405 * _S406;
                float3  _S408 = make_float3 (_S407);
                float _S409 = 0.37317633628845215f * z_2;
                float _S410 = 3.0f * yy_4;
                float _S411 = _S388 - _S398 - _S410;
                float _S412 = _S409 * _S411;
                float3  _S413 = make_float3 (_S412);
                float _S414 = -0.4570457935333252f * x_7;
                float _S415 = _S414 * _S406;
                float3  _S416 = make_float3 (_S415);
                float _S417 = 1.44530570507049561f * z_2;
                float _S418 = _S417 * _S393;
                float3  _S419 = make_float3 (_S418);
                float _S420 = -0.59004360437393188f * x_7;
                float _S421 = xx_4 - _S410;
                float _S422 = _S420 * _S421;
                float3  _S423 = make_float3 (_S422);
                features_11 = features_13 + make_float3 (_S400) * _S326.primal_0[int(9)] + make_float3 (_S403) * _S326.primal_0[int(10)] + make_float3 (_S407) * _S326.primal_0[int(11)] + make_float3 (_S412) * _S326.primal_0[int(12)] + make_float3 (_S415) * _S326.primal_0[int(13)] + make_float3 (_S418) * _S326.primal_0[int(14)] + make_float3 (_S422) * _S326.primal_0[int(15)];
                _S330 = _S423;
                _S331 = _S326.primal_0[int(15)];
                _S360 = _S420;
                _S361 = _S421;
                _S332 = _S419;
                _S333 = _S326.primal_0[int(14)];
                _S362 = _S417;
                _S334 = _S416;
                _S335 = _S326.primal_0[int(13)];
                _S363 = _S414;
                _S364 = _S406;
                _S336 = _S413;
                _S337 = _S326.primal_0[int(12)];
                _S365 = _S409;
                _S366 = _S411;
                _S338 = _S408;
                _S339 = _S326.primal_0[int(11)];
                _S367 = _S405;
                _S340 = _S404;
                _S341 = _S326.primal_0[int(10)];
                _S368 = _S402;
                _S342 = _S401;
                _S343 = _S326.primal_0[int(9)];
                _S369 = _S397;
                _S370 = _S399;
            }
            else
            {
                features_11 = features_13;
                _S330 = _S328;
                _S331 = _S328;
                _S360 = 0.0f;
                _S361 = 0.0f;
                _S332 = _S328;
                _S333 = _S328;
                _S362 = 0.0f;
                _S334 = _S328;
                _S335 = _S328;
                _S363 = 0.0f;
                _S364 = 0.0f;
                _S336 = _S328;
                _S337 = _S328;
                _S365 = 0.0f;
                _S366 = 0.0f;
                _S338 = _S328;
                _S339 = _S328;
                _S367 = 0.0f;
                _S340 = _S328;
                _S341 = _S328;
                _S368 = 0.0f;
                _S342 = _S328;
                _S343 = _S328;
                _S369 = 0.0f;
                _S370 = 0.0f;
            }
            float _S424 = _S363;
            float _S425 = _S364;
            float _S426 = _S365;
            float _S427 = _S366;
            float _S428 = _S367;
            float _S429 = _S368;
            float _S430 = _S369;
            float _S431 = _S370;
            _S375 = _S396;
            _S363 = _S393;
            _S364 = _S424;
            _S365 = _S425;
            _S366 = _S426;
            _S367 = _S427;
            _S368 = _S428;
            _S369 = _S429;
            _S370 = _S430;
            _S371 = _S431;
            _S344 = _S395;
            _S345 = _S326.primal_0[int(8)];
            _S346 = _S392;
            _S347 = _S326.primal_0[int(7)];
            _S348 = _S390;
            _S349 = _S326.primal_0[int(6)];
            _S350 = _S387;
            _S351 = _S326.primal_0[int(5)];
            _S352 = _S385;
            _S353 = _S326.primal_0[int(4)];
        }
        else
        {
            features_11 = features_12;
            _S375 = false;
            _S330 = _S328;
            _S331 = _S328;
            _S360 = 0.0f;
            _S361 = 0.0f;
            _S332 = _S328;
            _S333 = _S328;
            _S362 = 0.0f;
            _S363 = 0.0f;
            _S334 = _S328;
            _S335 = _S328;
            _S364 = 0.0f;
            _S365 = 0.0f;
            _S336 = _S328;
            _S337 = _S328;
            _S366 = 0.0f;
            _S367 = 0.0f;
            _S338 = _S328;
            _S339 = _S328;
            _S368 = 0.0f;
            _S340 = _S328;
            _S341 = _S328;
            _S369 = 0.0f;
            _S342 = _S328;
            _S343 = _S328;
            _S370 = 0.0f;
            _S371 = 0.0f;
            _S344 = _S328;
            _S345 = _S328;
            _S346 = _S328;
            _S347 = _S328;
            _S348 = _S328;
            _S349 = _S328;
            _S350 = _S328;
            _S351 = _S328;
            _S352 = _S328;
            _S353 = _S328;
        }
        bool _S432 = _S375;
        float _S433 = _S370;
        float _S434 = _S371;
        _S375 = _S383;
        _S376 = _S432;
        _S370 = z_2;
        _S371 = _S433;
        _S372 = _S434;
        _S373 = x_7;
        _S374 = y_4;
        _S354 = _S382;
        _S355 = _S326.primal_0[int(3)];
        _S356 = _S380;
        _S357 = _S326.primal_0[int(2)];
        _S358 = _S378;
        _S359 = _S326.primal_0[int(1)];
    }
    else
    {
        features_11 = features_10;
        _S375 = false;
        _S376 = false;
        _S330 = _S328;
        _S331 = _S328;
        _S360 = 0.0f;
        _S361 = 0.0f;
        _S332 = _S328;
        _S333 = _S328;
        _S362 = 0.0f;
        _S363 = 0.0f;
        _S334 = _S328;
        _S335 = _S328;
        _S364 = 0.0f;
        _S365 = 0.0f;
        _S336 = _S328;
        _S337 = _S328;
        _S366 = 0.0f;
        _S367 = 0.0f;
        _S338 = _S328;
        _S339 = _S328;
        _S368 = 0.0f;
        _S340 = _S328;
        _S341 = _S328;
        _S369 = 0.0f;
        _S370 = 0.0f;
        _S342 = _S328;
        _S343 = _S328;
        _S371 = 0.0f;
        _S372 = 0.0f;
        _S344 = _S328;
        _S345 = _S328;
        _S346 = _S328;
        _S347 = _S328;
        _S348 = _S328;
        _S349 = _S328;
        _S350 = _S328;
        _S351 = _S328;
        _S352 = _S328;
        _S353 = _S328;
        _S373 = 0.0f;
        _S374 = 0.0f;
        _S354 = _S328;
        _S355 = _S328;
        _S356 = _S328;
        _S357 = _S328;
        _S358 = _S328;
        _S359 = _S328;
    }
    float3  _S435 = features_11 + make_float3 (0.5f);
    float3  _S436 = make_float3 (0.0f);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S437;
    (&_S437)->primal_0 = _S435;
    (&_S437)->differential_0 = _S328;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S438;
    (&_S438)->primal_0 = _S436;
    (&_S438)->differential_0 = _S328;
    s_bwd_prop_max_0(&_S437, &_S438, _s_dOut_3);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S439 = _S437;
    FixedArray<float3 , 16>  _S440;
    if(_S329)
    {
        if(_S375)
        {
            if(_S376)
            {
                float3  _S441 = _S330 * _S439.differential_0;
                float3  _S442 = _S331 * _S439.differential_0;
                float _S443 = _S442.x + _S442.y + _S442.z;
                float _S444 = _S360 * _S443;
                float3  _S445 = _S332 * _S439.differential_0;
                float3  _S446 = _S333 * _S439.differential_0;
                float _S447 = _S446.x + _S446.y + _S446.z;
                float _S448 = _S362 * _S447;
                float _S449 = 1.44530570507049561f * (_S363 * _S447);
                float3  _S450 = _S334 * _S439.differential_0;
                float3  _S451 = _S335 * _S439.differential_0;
                float _S452 = _S451.x + _S451.y + _S451.z;
                float3  _S453 = _S336 * _S439.differential_0;
                float3  _S454 = _S337 * _S439.differential_0;
                float _S455 = _S454.x + _S454.y + _S454.z;
                float _S456 = _S366 * _S455;
                float _S457 = - _S456;
                float _S458 = 3.0f * (- _S444 + _S457);
                float _S459 = 0.37317633628845215f * (_S367 * _S455);
                float3  _S460 = _S338 * _S439.differential_0;
                float3  _S461 = _S339 * _S439.differential_0;
                float _S462 = _S461.x + _S461.y + _S461.z;
                float _S463 = _S364 * _S452 + _S368 * _S462;
                float _S464 = - _S463;
                float _S465 = 4.0f * _S463;
                float _S466 = -0.4570457935333252f * (_S365 * _S462);
                float3  _S467 = _S340 * _S439.differential_0;
                float3  _S468 = _S341 * _S439.differential_0;
                float _S469 = _S468.x + _S468.y + _S468.z;
                float _S470 = _S369 * _S469;
                float _S471 = 2.89061141014099121f * (_S370 * _S469);
                float3  _S472 = _S342 * _S439.differential_0;
                float3  _S473 = _S343 * _S439.differential_0;
                float _S474 = _S473.x + _S473.y + _S473.z;
                float _S475 = _S371 * _S474;
                float _S476 = - _S475;
                float _S477 = 3.0f * (_S457 + _S475);
                float _S478 = -0.59004360437393188f * (_S372 * _S474);
                float _S479 = -0.59004360437393188f * (_S361 * _S443) + -0.4570457935333252f * (_S365 * _S452);
                FixedArray<float3 , 16>  _S480;
                _S480[int(0)] = _S328;
                _S480[int(1)] = _S328;
                _S480[int(2)] = _S328;
                _S480[int(3)] = _S328;
                _S480[int(4)] = _S328;
                _S480[int(5)] = _S328;
                _S480[int(6)] = _S328;
                _S480[int(7)] = _S328;
                _S480[int(8)] = _S328;
                _S480[int(9)] = _S328;
                _S480[int(10)] = _S328;
                _S480[int(11)] = _S328;
                _S480[int(12)] = _S328;
                _S480[int(13)] = _S328;
                _S480[int(14)] = _S328;
                _S480[int(15)] = _S328;
                _S480[int(15)] = _S441;
                _S480[int(14)] = _S445;
                _S480[int(13)] = _S450;
                _S480[int(12)] = _S453;
                _S480[int(11)] = _S460;
                _S480[int(10)] = _S467;
                _S480[int(9)] = _S472;
                float _S481 = _S458 + _S464 + _S476;
                float _S482 = _S444 + _S464 + _S477;
                float _S483 = _S449 + _S459 + _S470;
                float _S484 = _S466 + _S478;
                _S360 = _S448;
                _S361 = _S456;
                _S362 = _S471;
                _S363 = _S465;
                _S364 = _S481;
                _S365 = _S482;
                _S440[int(0)] = _S480[int(0)];
                _S440[int(1)] = _S480[int(1)];
                _S440[int(2)] = _S480[int(2)];
                _S440[int(3)] = _S480[int(3)];
                _S440[int(4)] = _S480[int(4)];
                _S440[int(5)] = _S480[int(5)];
                _S440[int(6)] = _S480[int(6)];
                _S440[int(7)] = _S480[int(7)];
                _S440[int(8)] = _S480[int(8)];
                _S440[int(9)] = _S480[int(9)];
                _S440[int(10)] = _S480[int(10)];
                _S440[int(11)] = _S480[int(11)];
                _S440[int(12)] = _S480[int(12)];
                _S440[int(13)] = _S480[int(13)];
                _S440[int(14)] = _S480[int(14)];
                _S440[int(15)] = _S480[int(15)];
                _S366 = _S479;
                _S367 = _S484;
                _S368 = _S483;
            }
            else
            {
                _S360 = 0.0f;
                _S361 = 0.0f;
                _S362 = 0.0f;
                _S363 = 0.0f;
                _S364 = 0.0f;
                _S365 = 0.0f;
                _S440[int(0)] = _S328;
                _S440[int(1)] = _S328;
                _S440[int(2)] = _S328;
                _S440[int(3)] = _S328;
                _S440[int(4)] = _S328;
                _S440[int(5)] = _S328;
                _S440[int(6)] = _S328;
                _S440[int(7)] = _S328;
                _S440[int(8)] = _S328;
                _S440[int(9)] = _S328;
                _S440[int(10)] = _S328;
                _S440[int(11)] = _S328;
                _S440[int(12)] = _S328;
                _S440[int(13)] = _S328;
                _S440[int(14)] = _S328;
                _S440[int(15)] = _S328;
                _S366 = 0.0f;
                _S367 = 0.0f;
                _S368 = 0.0f;
            }
            float3  _S485 = _S344 * _S439.differential_0;
            float3  _S486 = _S345 * _S439.differential_0;
            float _S487 = 0.54627424478530884f * (_S486.x + _S486.y + _S486.z) + _S360;
            float3  _S488 = _S346 * _S439.differential_0;
            float3  _S489 = _S347 * _S439.differential_0;
            float s_diff_xz_T_0 = -1.09254848957061768f * (_S489.x + _S489.y + _S489.z);
            float3  _S490 = _S348 * _S439.differential_0;
            float3  _S491 = _S349 * _S439.differential_0;
            float _S492 = 0.31539157032966614f * (_S491.x + _S491.y + _S491.z);
            float _S493 = - _S492;
            float3  _S494 = _S350 * _S439.differential_0;
            float3  _S495 = _S351 * _S439.differential_0;
            float s_diff_yz_T_0 = -1.09254848957061768f * (_S495.x + _S495.y + _S495.z);
            float3  _S496 = _S352 * _S439.differential_0;
            float3  _S497 = _S353 * _S439.differential_0;
            float _S498 = _S373 * s_diff_xz_T_0;
            float _S499 = _S370 * s_diff_xz_T_0;
            float _S500 = _S374 * s_diff_yz_T_0;
            float _S501 = _S370 * s_diff_yz_T_0;
            float _S502 = 1.09254848957061768f * (_S497.x + _S497.y + _S497.z) + _S362;
            float _S503 = _S373 * _S502;
            float _S504 = _S374 * _S502;
            float _S505 = 2.0f * (_S492 + _S361) + _S363;
            float _S506 = _S370 * _S505;
            float _S507 = _S370 * _S505;
            float _S508 = - _S487 + _S493 + _S364;
            float _S509 = _S374 * _S508;
            float _S510 = _S374 * _S508;
            float _S511 = _S487 + _S493 + _S365;
            float _S512 = _S373 * _S511;
            float _S513 = _S373 * _S511;
            FixedArray<float3 , 16>  _S514;
            _S514[int(0)] = _S328;
            _S514[int(1)] = _S328;
            _S514[int(2)] = _S328;
            _S514[int(3)] = _S328;
            _S514[int(4)] = _S328;
            _S514[int(5)] = _S328;
            _S514[int(6)] = _S328;
            _S514[int(7)] = _S328;
            _S514[int(8)] = _S328;
            _S514[int(9)] = _S328;
            _S514[int(10)] = _S328;
            _S514[int(11)] = _S328;
            _S514[int(12)] = _S328;
            _S514[int(13)] = _S328;
            _S514[int(14)] = _S328;
            _S514[int(15)] = _S328;
            _S514[int(8)] = _S485;
            _S514[int(7)] = _S488;
            _S514[int(6)] = _S490;
            _S514[int(5)] = _S494;
            _S514[int(4)] = _S496;
            float3  _S515 = _S440[int(0)] + _S514[int(0)];
            float3  _S516 = _S440[int(1)] + _S514[int(1)];
            float3  _S517 = _S440[int(2)] + _S514[int(2)];
            float3  _S518 = _S440[int(3)] + _S514[int(3)];
            float3  _S519 = _S440[int(4)] + _S514[int(4)];
            float3  _S520 = _S440[int(5)] + _S514[int(5)];
            float3  _S521 = _S440[int(6)] + _S514[int(6)];
            float3  _S522 = _S440[int(7)] + _S514[int(7)];
            float3  _S523 = _S440[int(8)] + _S514[int(8)];
            float3  _S524 = _S440[int(9)] + _S514[int(9)];
            float3  _S525 = _S440[int(10)] + _S514[int(10)];
            float3  _S526 = _S440[int(11)] + _S514[int(11)];
            float3  _S527 = _S440[int(12)] + _S514[int(12)];
            float3  _S528 = _S440[int(13)] + _S514[int(13)];
            float3  _S529 = _S440[int(14)] + _S514[int(14)];
            float3  _S530 = _S440[int(15)] + _S514[int(15)];
            float _S531 = _S499 + _S504 + _S512 + _S513 + _S366;
            float _S532 = _S501 + _S503 + _S509 + _S510 + _S367;
            _S360 = _S498 + _S500 + _S506 + _S507 + _S368;
            _S361 = _S532;
            _S362 = _S531;
            _S440[int(0)] = _S515;
            _S440[int(1)] = _S516;
            _S440[int(2)] = _S517;
            _S440[int(3)] = _S518;
            _S440[int(4)] = _S519;
            _S440[int(5)] = _S520;
            _S440[int(6)] = _S521;
            _S440[int(7)] = _S522;
            _S440[int(8)] = _S523;
            _S440[int(9)] = _S524;
            _S440[int(10)] = _S525;
            _S440[int(11)] = _S526;
            _S440[int(12)] = _S527;
            _S440[int(13)] = _S528;
            _S440[int(14)] = _S529;
            _S440[int(15)] = _S530;
        }
        else
        {
            _S360 = 0.0f;
            _S361 = 0.0f;
            _S362 = 0.0f;
            _S440[int(0)] = _S328;
            _S440[int(1)] = _S328;
            _S440[int(2)] = _S328;
            _S440[int(3)] = _S328;
            _S440[int(4)] = _S328;
            _S440[int(5)] = _S328;
            _S440[int(6)] = _S328;
            _S440[int(7)] = _S328;
            _S440[int(8)] = _S328;
            _S440[int(9)] = _S328;
            _S440[int(10)] = _S328;
            _S440[int(11)] = _S328;
            _S440[int(12)] = _S328;
            _S440[int(13)] = _S328;
            _S440[int(14)] = _S328;
            _S440[int(15)] = _S328;
        }
        float3  _S533 = - _S439.differential_0;
        float3  _S534 = _S354 * _S533;
        float3  _S535 = _S355 * _S533;
        float3  _S536 = _S356 * _S439.differential_0;
        float3  _S537 = _S357 * _S439.differential_0;
        float3  _S538 = _S358 * _S533;
        float3  _S539 = _S359 * _S533;
        float _S540 = 0.48860251903533936f * (_S537.x + _S537.y + _S537.z) + _S360;
        float _S541 = 0.48860251903533936f * (_S539.x + _S539.y + _S539.z) + _S361;
        float _S542 = 0.48860251903533936f * (_S535.x + _S535.y + _S535.z) + _S362;
        FixedArray<float3 , 16>  _S543;
        _S543[int(0)] = _S328;
        _S543[int(1)] = _S328;
        _S543[int(2)] = _S328;
        _S543[int(3)] = _S328;
        _S543[int(4)] = _S328;
        _S543[int(5)] = _S328;
        _S543[int(6)] = _S328;
        _S543[int(7)] = _S328;
        _S543[int(8)] = _S328;
        _S543[int(9)] = _S328;
        _S543[int(10)] = _S328;
        _S543[int(11)] = _S328;
        _S543[int(12)] = _S328;
        _S543[int(13)] = _S328;
        _S543[int(14)] = _S328;
        _S543[int(15)] = _S328;
        _S543[int(3)] = _S534;
        _S543[int(2)] = _S536;
        _S543[int(1)] = _S538;
        float3  _S544 = _S440[int(0)] + _S543[int(0)];
        float3  _S545 = _S440[int(1)] + _S543[int(1)];
        float3  _S546 = _S440[int(2)] + _S543[int(2)];
        float3  _S547 = _S440[int(3)] + _S543[int(3)];
        float3  _S548 = _S440[int(4)] + _S543[int(4)];
        float3  _S549 = _S440[int(5)] + _S543[int(5)];
        float3  _S550 = _S440[int(6)] + _S543[int(6)];
        float3  _S551 = _S440[int(7)] + _S543[int(7)];
        float3  _S552 = _S440[int(8)] + _S543[int(8)];
        float3  _S553 = _S440[int(9)] + _S543[int(9)];
        float3  _S554 = _S440[int(10)] + _S543[int(10)];
        float3  _S555 = _S440[int(11)] + _S543[int(11)];
        float3  _S556 = _S440[int(12)] + _S543[int(12)];
        float3  _S557 = _S440[int(13)] + _S543[int(13)];
        float3  _S558 = _S440[int(14)] + _S543[int(14)];
        float3  _S559 = _S440[int(15)] + _S543[int(15)];
        features_11 = make_float3 (_S542, _S541, _S540);
        _S440[int(0)] = _S544;
        _S440[int(1)] = _S545;
        _S440[int(2)] = _S546;
        _S440[int(3)] = _S547;
        _S440[int(4)] = _S548;
        _S440[int(5)] = _S549;
        _S440[int(6)] = _S550;
        _S440[int(7)] = _S551;
        _S440[int(8)] = _S552;
        _S440[int(9)] = _S553;
        _S440[int(10)] = _S554;
        _S440[int(11)] = _S555;
        _S440[int(12)] = _S556;
        _S440[int(13)] = _S557;
        _S440[int(14)] = _S558;
        _S440[int(15)] = _S559;
    }
    else
    {
        features_11 = _S328;
        _S440[int(0)] = _S328;
        _S440[int(1)] = _S328;
        _S440[int(2)] = _S328;
        _S440[int(3)] = _S328;
        _S440[int(4)] = _S328;
        _S440[int(5)] = _S328;
        _S440[int(6)] = _S328;
        _S440[int(7)] = _S328;
        _S440[int(8)] = _S328;
        _S440[int(9)] = _S328;
        _S440[int(10)] = _S328;
        _S440[int(11)] = _S328;
        _S440[int(12)] = _S328;
        _S440[int(13)] = _S328;
        _S440[int(14)] = _S328;
        _S440[int(15)] = _S328;
    }
    float3  _S560 = make_float3 (0.282094806432724f) * _S439.differential_0;
    dpdirection_1->primal_0 = (*dpdirection_1).primal_0;
    dpdirection_1->differential_0 = features_11;
    FixedArray<float3 , 16>  _S561;
    _S561[int(0)] = _S328;
    _S561[int(1)] = _S328;
    _S561[int(2)] = _S328;
    _S561[int(3)] = _S328;
    _S561[int(4)] = _S328;
    _S561[int(5)] = _S328;
    _S561[int(6)] = _S328;
    _S561[int(7)] = _S328;
    _S561[int(8)] = _S328;
    _S561[int(9)] = _S328;
    _S561[int(10)] = _S328;
    _S561[int(11)] = _S328;
    _S561[int(12)] = _S328;
    _S561[int(13)] = _S328;
    _S561[int(14)] = _S328;
    _S561[int(15)] = _S328;
    _S561[int(0)] = _S560;
    FixedArray<float3 , 16>  _S562 = {
        _S440[int(0)] + _S561[int(0)], _S440[int(1)] + _S561[int(1)], _S440[int(2)] + _S561[int(2)], _S440[int(3)] + _S561[int(3)], _S440[int(4)] + _S561[int(4)], _S440[int(5)] + _S561[int(5)], _S440[int(6)] + _S561[int(6)], _S440[int(7)] + _S561[int(7)], _S440[int(8)] + _S561[int(8)], _S440[int(9)] + _S561[int(9)], _S440[int(10)] + _S561[int(10)], _S440[int(11)] + _S561[int(11)], _S440[int(12)] + _S561[int(12)], _S440[int(13)] + _S561[int(13)], _S440[int(14)] + _S561[int(14)], _S440[int(15)] + _S561[int(15)]
    };
    dpcoefficients_1->primal_0 = (*dpcoefficients_1).primal_0;
    dpcoefficients_1->differential_0 = _S562;
    return;
}

__device__ void particleFeaturesIntegrateBwdToBuffer(float3  incidentDirection_2, float alpha_5, float * alphaGrad_2, uint particleIdx_7, shRadiativeParticle_CommonParameters_0 commonParameters_7, float3  features_14, float3  * integratedFeatures_3, float3  * integratedFeaturesGrad_1)
{
    shRadiativeParticle_Parameters_0 _S563;
    bool _S564 = alpha_5 > 0.0f;
    if(_S564)
    {
        int i_6;
        float3  _S565 = (*integratedFeatures_3 - features_14 * make_float3 (alpha_5)) * make_float3 (1.0f / (1.0f - alpha_5));
        *integratedFeatures_3 = _S565;
        float3  _S566 = *integratedFeaturesGrad_1;
        float3  _S567 = make_float3 (0.0f);
        FixedArray<float3 , 16>  _S568 = {
            _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567
        };
        shRadiativeParticle_Parameters_0 _S569 = { _S568 };
        s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0 _S570;
        uint _S571 = particleIdx_7 * 16U;
        int _S572 = int(uint(commonParameters_7.sphDegree_0));
        shRadiativeParticle_Parameters_0 _S573 = shRadiativeParticle_Parameters_x24_syn_dzero_0();
        float3  _S574 = make_float3 (alpha_5);
        FixedArray<float3 , 16>  _S575 = { _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567, _S567 };
        for(;;)
        {
            (&_S570)->_S314 = _S567;
            (&_S570)->_S315 = _S569;
            (&_S570)->_S314 = _S567;
            (&(&_S570)->_S315)->sphCoefficients_0 = _S568;
            (&_S570)->_S314 = _S565;
            for(;;)
            {
                i_6 = int(0);
                shRadiativeParticle_Parameters_0 parameters_4;
                #pragma unroll
                for(;;)
                {
                    if(i_6 < int(16))
                    {
                    }
                    else
                    {
                        break;
                    }
                    (&parameters_4)->sphCoefficients_0[i_6] = *(commonParameters_7.parametersBuffer_1._dataPtr_1 + (_S571 + uint(i_6)));
                    i_6 = i_6 + int(1);
                }
                _S563 = parameters_4;
                break;
            }
            (&_S570)->_S315 = _S563;
            break;
        }
        float _S576;
        float3  dpintegratedRadiance_1;
        s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0 _S577 = _S570;
        float3  _S578 = s_primal_ctx_sphericalHarmonics_decode_0(_S572, _S570._S315.sphCoefficients_0, incidentDirection_2);
        for(;;)
        {
            float3  _S579;
            FixedArray<float3 , 16>  _S580;
            if(_S564)
            {
                if(_S564)
                {
                    _S579 = _S574;
                }
                else
                {
                    _S579 = _S567;
                }
                float3  _S581 = _S579;
                _S579 = _S578;
                dpintegratedRadiance_1 = _S581;
                _S580 = _S577._S315.sphCoefficients_0;
                i_6 = _S572;
            }
            else
            {
                _S579 = _S567;
                dpintegratedRadiance_1 = _S567;
                _S580[int(0)] = _S567;
                _S580[int(1)] = _S567;
                _S580[int(2)] = _S567;
                _S580[int(3)] = _S567;
                _S580[int(4)] = _S567;
                _S580[int(5)] = _S567;
                _S580[int(6)] = _S567;
                _S580[int(7)] = _S567;
                _S580[int(8)] = _S567;
                _S580[int(9)] = _S567;
                _S580[int(10)] = _S567;
                _S580[int(11)] = _S567;
                _S580[int(12)] = _S567;
                _S580[int(13)] = _S567;
                _S580[int(14)] = _S567;
                _S580[int(15)] = _S567;
                i_6 = int(0);
            }
            shRadiativeParticle_Parameters_0 _S582;
            if(_S564)
            {
                if(_S564)
                {
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S583;
                    (&_S583)->primal_0 = _S577._S314;
                    (&_S583)->differential_0 = _S567;
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S584;
                    (&_S584)->primal_0 = _S579;
                    (&_S584)->differential_0 = _S567;
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S585;
                    (&_S585)->primal_0 = dpintegratedRadiance_1;
                    (&_S585)->differential_0 = _S567;
                    s_bwd_prop_lerp_0(&_S583, &_S584, &_S585, _S566);
                    float _S586 = _S585.differential_0.x + _S585.differential_0.y + _S585.differential_0.z;
                    _S579 = _S584.differential_0;
                    dpintegratedRadiance_1 = _S583.differential_0;
                    _S576 = _S586;
                }
                else
                {
                    _S579 = _S567;
                    dpintegratedRadiance_1 = _S566;
                    _S576 = 0.0f;
                }
                DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S587;
                (&_S587)->primal_0 = _S580;
                (&_S587)->differential_0 = _S575;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S588;
                (&_S588)->primal_0 = incidentDirection_2;
                (&_S588)->differential_0 = _S567;
                s_bwd_prop_sphericalHarmonics_decode_0(i_6, &_S587, &_S588, _S579);
                shRadiativeParticle_Parameters_0 _S589 = _S573;
                (&_S589)->sphCoefficients_0 = _S587.differential_0;
                _S582 = shRadiativeParticle_Parameters_x24_syn_dadd_0(_S573, _S589);
            }
            else
            {
                _S582 = _S573;
                dpintegratedRadiance_1 = _S566;
                _S576 = 0.0f;
            }
            for(;;)
            {
                i_6 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(i_6 < int(16))
                    {
                    }
                    else
                    {
                        break;
                    }
                    shRadiativeParticle_Parameters_0 _S590 = _S582;
                    int _S591 = i_6;
                    int j_2;
                    if(commonParameters_7.parametersBuffer_1.exclusiveGradient_1)
                    {
                        j_2 = int(0);
                        #pragma unroll
                        for(;;)
                        {
                            if(j_2 < int(3))
                            {
                            }
                            else
                            {
                                break;
                            }
                            *_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S571 + uint(i_6)), j_2) = *_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S571 + uint(i_6)), j_2) + _slang_vector_get_element(_S590.sphCoefficients_0[_S591], j_2);
                            j_2 = j_2 + int(1);
                        }
                    }
                    else
                    {
                        j_2 = int(0);
                        #pragma unroll
                        for(;;)
                        {
                            if(j_2 < int(3))
                            {
                            }
                            else
                            {
                                break;
                            }
                            float _S592 = atomicAdd(_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S571 + uint(i_6)), j_2), _slang_vector_get_element(_S590.sphCoefficients_0[_S591], j_2));
                            j_2 = j_2 + int(1);
                        }
                    }
                    i_6 = i_6 + int(1);
                }
                break;
            }
            break;
        }
        *integratedFeaturesGrad_1 = dpintegratedRadiance_1;
        *alphaGrad_2 = _S576;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0
{
    shRadiativeParticle_Parameters_0 _S593;
};

__device__ void particleFeaturesBwdToBuffer(uint particleIdx_8, shRadiativeParticle_CommonParameters_0 commonParameters_8, float3  featuresGrad_1, float3  incidentDirection_3)
{
    int i_7;
    shRadiativeParticle_Parameters_0 _S594;
    uint _S595 = uint(commonParameters_8.sphDegree_0);
    float3  _S596 = make_float3 (0.0f);
    FixedArray<float3 , 16>  _S597 = {
        _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596
    };
    shRadiativeParticle_Parameters_0 _S598 = { _S597 };
    s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0 _S599;
    (&_S599)->_S593 = _S598;
    (&(&_S599)->_S593)->sphCoefficients_0 = _S597;
    uint _S600 = particleIdx_8 * 16U;
    int _S601 = int(_S595);
    FixedArray<float3 , 16>  _S602 = { _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596, _S596 };
    shRadiativeParticle_Parameters_0 _S603 = shRadiativeParticle_Parameters_x24_syn_dzero_0();
    for(;;)
    {
        i_7 = int(0);
        shRadiativeParticle_Parameters_0 parameters_5;
        #pragma unroll
        for(;;)
        {
            if(i_7 < int(16))
            {
            }
            else
            {
                break;
            }
            (&parameters_5)->sphCoefficients_0[i_7] = *(commonParameters_8.parametersBuffer_1._dataPtr_1 + (_S600 + uint(i_7)));
            i_7 = i_7 + int(1);
        }
        _S594 = parameters_5;
        break;
    }
    (&_S599)->_S593 = _S594;
    DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S604;
    (&_S604)->primal_0 = _S599._S593.sphCoefficients_0;
    (&_S604)->differential_0 = _S602;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S605;
    (&_S605)->primal_0 = incidentDirection_3;
    (&_S605)->differential_0 = _S596;
    s_bwd_prop_sphericalHarmonics_decode_0(_S601, &_S604, &_S605, featuresGrad_1);
    shRadiativeParticle_Parameters_0 _S606 = _S603;
    (&_S606)->sphCoefficients_0 = _S604.differential_0;
    shRadiativeParticle_Parameters_0 _S607 = _S606;
    for(;;)
    {
        i_7 = int(0);
        #pragma unroll
        for(;;)
        {
            if(i_7 < int(16))
            {
            }
            else
            {
                break;
            }
            int _S608 = i_7;
            int j_3;
            if(commonParameters_8.parametersBuffer_1.exclusiveGradient_1)
            {
                j_3 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(j_3 < int(3))
                    {
                    }
                    else
                    {
                        break;
                    }
                    *_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S600 + uint(i_7)), j_3) = *_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S600 + uint(i_7)), j_3) + _slang_vector_get_element(_S607.sphCoefficients_0[_S608], j_3);
                    j_3 = j_3 + int(1);
                }
            }
            else
            {
                j_3 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(j_3 < int(3))
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S609 = atomicAdd(_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S600 + uint(i_7)), j_3), _slang_vector_get_element(_S607.sphCoefficients_0[_S608], j_3));
                    j_3 = j_3 + int(1);
                }
            }
            i_7 = i_7 + int(1);
        }
        break;
    }
    return;
}

