#include "keyGenAndCacheStore.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unistd.h>

/** Zipf-like random distribution.
 *
 * "Rejection-inversion to generate variates from monotone discrete
 * distributions", Wolfgang Hörmann and Gerhard Derflinger
 * ACM TOMACS 6.3 (1996): 169-184
 */
template<class IntType = unsigned long, class RealType = double>
class zipf_distribution
{
public:
    typedef RealType input_type;
    typedef IntType result_type;

    static_assert(std::numeric_limits<IntType>::is_integer, "");
    static_assert(!std::numeric_limits<RealType>::is_integer, "");

    zipf_distribution(const IntType n=std::numeric_limits<IntType>::max(),
                      const RealType q=0.99)
        : n(n)
        , q(q)
        , H_x1(H(1.5) - 1.0)
        , H_n(H(n + 0.5))
        , dist(H_x1, H_n)
    {}

    IntType operator()(std::mt19937& rng)
    {
        while (true) {
            const RealType u = dist(rng);
            const RealType x = H_inv(u);
            const IntType  k = clamp<IntType>(std::round(x), 1, n);
            if (u >= H(k + 0.5) - h(k)) {
                return k;
            }
        }
    }

private:
    /** Clamp x to [min, max]. */
    template<typename T>
    static constexpr T clamp(const T x, const T min, const T max)
    {
        return std::max(min, std::min(max, x));
    }

    /** exp(x) - 1 / x */
    static double
    expxm1bx(const double x)
    {
        return (std::abs(x) > epsilon)
            ? std::expm1(x) / x
            : (1.0 + x/2.0 * (1.0 + x/3.0 * (1.0 + x/4.0)));
    }

    /** H(x) = log(x) if q == 1, (x^(1-q) - 1)/(1 - q) otherwise.
     * H(x) is an integral of h(x).
     *
     * Note the numerator is one less than in the paper order to work with all
     * positive q.
     */
    const RealType H(const RealType x)
    {
        const RealType log_x = std::log(x);
        return expxm1bx((1.0 - q) * log_x) * log_x;
    }

    /** log(1 + x) / x */
    static RealType
    log1pxbx(const RealType x)
    {
        return (std::abs(x) > epsilon)
            ? std::log1p(x) / x
            : 1.0 - x * ((1/2.0) - x * ((1/3.0) - x * (1/4.0)));
    }

    /** The inverse function of H(x) */
    const RealType H_inv(const RealType x)
    {
        const RealType t = std::max(-1.0, x * (1.0 - q));
        return std::exp(log1pxbx(t) * x);
    }

    /** That hat function h(x) = 1 / (x ^ q) */
    const RealType h(const RealType x)
    {
        return std::exp(-q * std::log(x));
    }

    static constexpr RealType epsilon = 1e-8;

    IntType                                  n;     ///< Number of elements
    RealType                                 q;     ///< Exponent
    RealType                                 H_x1;  ///< H(x_1)
    RealType                                 H_n;   ///< H(n)
    std::uniform_real_distribution<RealType> dist;  ///< [H(x_1), H(n)]
};

using namespace std;

static mt19937 gen;
// 1st item is generated 10% of the times
static zipf_distribution<int, double> *dist = nullptr;

void zipf_setup(unsigned long nbItems, double param)
{
  if (dist != nullptr) delete dist;
  dist = new zipf_distribution<int, double>(nbItems, param);
}

unsigned long zipf_gen()
{
  return (*dist)(gen);
}

void fill_array_with_items(int nbItems, unsigned long itemSpace, double param, void(*callback)(int idx, int total, long long item))
{
    char filename[1024];
    FILE *fp;

    sprintf(filename, "zipf_key_gen_I%i_P%lli.tsv", nbItems, (long long)(param*1000));
    if (access(filename, F_OK) == 0) {
        // file exists
        fp = fopen(filename, "r");
        for (int i = 0; i < nbItems; ++i) {
            long long item;
            fscanf(fp, "%llx\n", &item); // TODO: must check if hit the end of file...
            callback(i, nbItems, item);
        }
        fclose(fp);
    } else {
        // file doesn't exist
        fp = fopen(filename, "w");
        zipf_setup(itemSpace, param);
        for (int i = 0; i < nbItems; ++i) {
            long long item = zipf_gen();
            fprintf(fp, "%llx\n", item);
            callback(i, nbItems, item);
        }
        fclose(fp);
    }
}

// returns 0 if file does not exist, returns -1 and exits if file exists 
int storeArray(const char *name, long nbItems, int sizeItem, void *values)
{
    char filename[1024];
    FILE *fp;

    sprintf(filename, "%s_I%li.bin", name, nbItems);
    if (access(filename, F_OK) == 0) {
        return -1;
    } else {
        fp = fopen(filename, "w");
        fwrite(values, sizeItem, nbItems, fp);
        fclose(fp);
        return 0;
    }
}

// returns -1 if file does not exist, returns 0 and loads if file exists 
int loadArray(const char *name, long nbItems, int sizeItem, void *values)
{
    char filename[1024];
    FILE *fp;

    sprintf(filename, "%s_I%li.bin", name, nbItems);
    if (access(filename, F_OK) == 0) {
        fp = fopen(filename, "r");
        fread(values, sizeItem, nbItems, fp);
        fclose(fp);
        return 0;
    } else {
        return -1;
    }
}
