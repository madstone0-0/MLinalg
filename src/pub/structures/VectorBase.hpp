#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Concepts.hpp"
#include "Aliases.hpp"
#include "Matrix.hpp"
#include "VectorOps.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr, mlinalg::structures::helpers::unwrap;

namespace mlinalg::structures {

    /**
     * @brief Base CTRP class for Vector operations
     *
     * @tparam D Derived class type, used for CRTP
     * @tparam number The type of the elements in the vector (e.g., float
     */
    template <typename D, Number number>
    class VectorBase {
       protected:
        // CRTP helpers
        D& d() { return static_cast<D&>(*this); }
        const D& d() const { return static_cast<const D&>(*this); }

        // Protected constructor - only derived classes can construct
        VectorBase() = default;
        ~VectorBase() = default;

       public:
        using value_type = number;
        using size_type = size_t;
        using ref = number&;
        using const_ref = const number&;

        // ======================
        // Indexing and Accessors
        // ======================

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        constexpr number& at(size_t i) { return vectorAt<number>(d().row, i); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return a reference to the ith element
         */
        constexpr number& operator[](size_t i) { return d().row[i]; }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        constexpr const number& at(size_t i) const { return vectorConstAt<number>(d().row, i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return The ith element
         */
        constexpr number& operator[](size_t i) const { return const_cast<number&>(d().row[i]); }

        // ============
        // Iteration
        // ============

        /**
         * @brief Begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector
         */
        constexpr auto begin() const { return d().row.begin(); }

        /**
         * @brief End iterator for the vector
         *
         * @return An iterator to the end of the vector
         */
        constexpr auto end() const { return d().row.end(); }

        /**
         * @brief Const begin iterator for the vector
         *
         * @return A const iterator to the beginning of the vector
         */
        constexpr auto cbegin() const { return d().row.cbegin(); }

        /**
         * @brief Const end iterator for the vector
         *
         * @return A c onst iterator to the end of the vector
         */
        constexpr auto cend() const { return d().row.cend(); }

        /**
         * @brief Last element of the vector
         *
         * @return A reference to the last element of the vector
         */
        constexpr auto& back() { return d().row.back(); }

        /**
         * @brief Const last element of the vector
         *
         * @return A const reference to the last element of the vector
         */
        constexpr auto& back() const { return d().row.back(); }

        /**
         * @brief Reverse begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector in reverse
         */
        constexpr auto rbegin() { return d().row.rbegin(); }

        /**
         * @brief Reverse end iterator for the vector
         *
         * @return An iterator to the end of the vector in reverse
         */
        constexpr auto rend() { return d().row.rend(); }

        // ============
        // Comparision
        // ============

        /**
         * @brief Equality operator
         *
         * @param other Vector to compare
         * @return true if all the entires in the vector are equal to all the entires in the other vector else false
         */
        template <typename OtherD>
        bool operator==(const VectorBase<OtherD, number>& other) const {
            return vectorEqual(d().row, static_cast<const OtherD&>(other).row);
        }

        // ======================
        // Arithmetic Operations
        // ======================

        /**
         * @brief Vector subtraction
         *
         * @param other the vector to subtract
         * @return the vector resulting from the subtraction
         */
        template <typename OtherD>
        D operator-(const VectorBase<OtherD, number>& other) const {
            auto res = d();
            res -= other;
            return static_cast<D&>(res);
        }

        /**
         * @brief Vector Negation
         *
         * @return the negeated vector
         */
        D& operator-() {
            vectorNeg<number>(d().row);
            return d();
        }

        /**
         * @brief In-place vector subtraction
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        template <typename OtherD>
        D& operator-=(const VectorBase<OtherD, number>& other) {
            vectorSubI<number>(d().row, static_cast<const OtherD&>(other).row);
            return d();
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        template <typename OtherD>
        D operator+(const VectorBase<OtherD, number>& other) const {
            auto res = d();
            res += other;
            return static_cast<D&>(res);
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        template <typename OtherD>
        D& operator+=(const VectorBase<OtherD, number>& other) {
            vectorAddI<number>(d().row, static_cast<const OtherD&>(other).row);
            return d();
        }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        D operator/(const number& scalar) const {
            auto res = d();
            res /= scalar;
            return static_cast<D&>(res);
        }

        /**
         * @brief In-place vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return
         */
        D& operator/=(const number& scalar) {
            vectorScalarDivI(d().row, scalar);
            return d();
        }

        /**
         * @brief In-place vector multiplication by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        D& operator*=(const number& scalar) {
            vectorScalarMultI<number>(d().row, scalar);
            return d();
        }

        /**
         * @brief Vector multiplication by a vector
         *
         * @param vec Another vector of the same size as the vector
         * @return  A 1x1 vector containing the dot product of the two vectors
         */
        template <typename OtherD, Number num>
        friend auto operator*(const VectorBase<D, num>& vec, const VectorBase<OtherD, num>& other) {
            return vectorVectorMult(vec.d(), static_cast<const OtherD&>(other));
        }

        /**
         * @brief Vector multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return the vector resulting from the multiplication
         */
        template <Number num>
        friend D operator*(const VectorBase<D, num>& vec, const num& scalar) {
            auto res = vec.d();
            res *= scalar;
            return static_cast<const D&>(res);
        }

        /**
         * @brief Vector multiplication by a vector
         *
         * @param vec Another vector of the same size as the vector
         * @return  A 1x1 vector containing the dot product of the two vectors
         */
        template <typename OtherD>
        auto operator*(const VectorBase<OtherD, number>& other) const {
            return vectorVectorMult(d(), static_cast<const OtherD&>(other));
        }

        // ======================
        // Vector Operations
        // ======================

        /**
         * @brief Find the dot product of this vector and another vector
         *
         * @param other The other vector
         * @return the dot product of the two vectors
         */
        template <typename OtherD>
        [[nodiscard]] double dot(const VectorBase<OtherD, number>& other) const {
            return vectorDot(d(), static_cast<const OtherD&>(other));
        }

        /**
         * @brief Apply a function to each element of the vector
         *
         * @tparam F Function type that takes a number and returns void
         * @param f Function to apply to each element of the vector
         * @return A reference to the same vector
         */
        template <typename F>
        D& apply(F f) {
            vectorApply(d().row, f);
            return d();
        }

        /**
         * @brief Apply a function to each element of the vector with another vector
         *
         * @tparam F Function type that takes two numbers and returns void
         * @param other The other vector to apply the function with
         * @param f Function to apply to each element of the vector
         * @return A reference to the same vector
         */
        template <typename OtherD, typename F>
        D& apply(const VectorBase<OtherD, number>& other, F f) {
            vectorApply(d().row, static_cast<const OtherD&>(other).row, f);
            return d();
        }

        /**
         * @brief Normalize a vector into a unit vector
         *
         * @return the normalized vector
         */
        D normalize() const { return vectorNormalize(d()); }

        /**
         * @brief Normalize a vector into a unit vector in-place
         *
         * @return A reference to the same vector
         */
        D& normalizeI() { return vectorNormalizeI(d()); }

        // ======================
        // Norms and Distances
        // ======================

        /**
         * @brief Find the length of the vector
         *
         * @return the length of the vector
         */
        [[nodiscard]] double length() const { return EuclideanNorm(d()); }

        /**
         * @brief Find the l2 norm of the vector
         *
         * @return the l2 norm of the vector
         */
        [[nodiscard]] double l2() const { return length(); }

        /**
         * @brief Find the euclidean norm of the vector
         *
         * This is the same as the length of the vector, and the L2 norm of the vector
         *
         * \f[
         * ||x||_2 = \sqrt{x^T \cdot x}
         * \f]
         *
         * @return the euclidean norm of the vector
         */
        [[nodiscard]] double euclid() const { return length(); }

        /**
         * @brief Find the L1 norm of the vector
         *
         * This is the same as the sum of the absolute values of the elements in the vector
         *
         * \f[
         * ||x||_1 = \sum{|x_i|}
         * \f]
         *
         * @return the L1 norm of the vector
         */
        [[nodiscard]] double l1() const { return L1Norm<number>(d().row); }

        /**
         * @brief Find the weighted L2 norm of the vector
         *
         * Each of the coordinates of a vector space is given a weight
         *
         * \f[
         * ||x||_W = \sqrt{\sum{w_i * x_i^2}}
         * \f]
         *
         * @param otherVec The other vector
         * @return the weighted L2 norm of the vector
         */
        template <typename OtherD>
        double weightedL2(const VectorBase<OtherD, number>& other) const {
            return WeightedL2Norm<number>(d().row, static_cast<const OtherD&>(other).row);
        }

        /**
         * @brief Find the distance between this vector and another vector
         *
         * @param other The other vector
         * @return the distance between the two vectors
         */
        template <typename OtherD>
        [[nodiscard]] double dist(const VectorBase<OtherD, number>& other) const {
            return vectorDist(d(), static_cast<const OtherD&>(other));
        }

        // ======================
        // Miscellaneous Operations
        // ======================
        /**
         * @brief Clear the vector, i.e. set all elements to zero
         */
        void clear() { vectorClear(d()); }

        const number* data() const noexcept { return d().row.data(); }

        number* data() noexcept { return d().row.data(); }

        explicit operator std::string() const { return vectorStringRepr(d().row); }

        friend std::ostream& operator<<(std::ostream& os, const VectorBase& vec) {
            os << std::string(vec.d());
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const optional<VectorBase>& rowPot) {
            if (!rowPot.has_value()) {
                os << "Empty Vector";
                return os;
            }

            return vectorOptionalRepr(os, rowPot.value().row);
        }
    };

    template <typename D, Number num>
    D operator*(const num& scalar, const VectorBase<D, num>& vec) {
        return static_cast<const D&>(vec) * scalar;
    }

}  // namespace mlinalg::structures
