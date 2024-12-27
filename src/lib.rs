/*
Copyright (c) 2023 Michał Margos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

//! An unordered, growable map type with heap-allocated contents, written as
//! `Registry<T>`. All contained values have their unique indices attributed
//! to them at the point of insertion. Indices are freed when the items are
//! removed.
//!
//! Registries have *O*(1) indexing, *O*(1) removal and *O*(1) insertion.
//!
//! # Examples
//!
//! You can explicitly create a [`Registry`] with [`Registry::new`]:
//!
//! ```
//! let r: Registry<i32> = Registry::new();
//! ```
//!
//! You can [`insert`] values into the registry:
//!
//! ```
//! let mut r = Registry::new();
//!
//! let index = r.insert(3);
//! ```
//!
//! Removing values works in much the same way:
//!
//! ```
//! let mut r = Registry::new();
//!
//! let index = r.insert(3);
//! let three = r.remove(index);
//! ```
//!
//! Registries also support indexing (through the [`Index`] and [`IndexMut`] traits):
//!
//! ```
//! let mut r = Registry::new();
//! let i1 = r.insert(1);
//! let i2 = r.insert(4);
//!
//! let one = r[i1];
//! r[i2] = r[i2] + 5;
//! ```
//!
//! [`insert`]: Registry::insert

#![no_std]

use core::{
    fmt::{self, Debug},
    iter::Enumerate,
    marker::PhantomData,
    ops::{Index, IndexMut},
    slice,
};

extern crate alloc;

use alloc::vec::{self, Vec};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Serialize,
};

/// An unordered, growable map type with heap-allocated contents, written as
/// `Registry<T>`.
///
/// # Examples
///
/// ```
/// let mut reg = Registry::new();
/// let one = reg.insert(1);
/// let two = reg.insert(2);
///
/// assert_eq!(reg.count(), 2);
/// assert_eq!(reg[one], 1);
///
/// assert_eq!(vec.remove(two), Some(2));
/// assert_eq!(reg.count(), 1);
///
/// reg[one] = 7;
/// assert_eq!(reg[one], 7);
///
/// for x in &reg {
///     println!("{x}");
/// }
/// ```
///
/// # Indexing
///
/// The `Registry` type allows access to values by index, because it implements the
/// [`Index`] trait. An example will be more explicit:
///
/// ```
/// let reg = Registry::new();
/// let three = reg.insert(3);
/// println!("{}", reg[three]); // it will display '3'
/// ```
///
/// However be careful: if you try to access an index which isn't in the `Registry`,
/// your software will panic! You cannot do this:
///
/// ```should_panic
/// let reg = Registry::new();
/// reg.insert(3);
/// println!("{}", reg[5]); // it will panic!
/// ```
///
/// Use [`get`] and [`get_mut`] if you want to check whether the index is in
/// the `Registry`.
///
/// # Capacity and reallocation
///
/// Registries under the hood are simply two `Vec` and so their memory management
/// behaves as such.
///
/// # Guarantees
///
/// Because of the way the `Registry` is implemented, most guarantees granted
/// by a Rust `Vec` are also granted by the `Registry`. There are, however, some
/// exceptions.
///
/// Most importantly, element removal might cause reallocation. Thus, the size of
/// allocated space is only guaranteed to be less or equal to the size allocated
/// by a vector, given the same insertions and removals.
///
/// Apart from that, `Registry` does not guarantee either continuity or order of
/// inserted data.
#[derive(Clone)]
pub struct Registry<T> {
    data: Vec<Option<T>>,
    indices: Vec<usize>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
        }
    }
}

impl<T> Registry<T> {
    /// Constructs a new, empty `Registry<T>`.
    ///
    /// The registry will not allocate until elements are inserted into it.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut reg: Registry<i32> = Registry::new();
    /// ```
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Constructs a new, empty `Registry<T>` with at least the specified capacity.
    ///
    /// The registry will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is 0, the registry will not allocate.
    ///
    /// It is important to note that although the returned registry has the
    /// minimum *capacity* specified, the registry will have a zero *count*. For
    /// an explanation of the difference between length and capacity, see
    /// *[Capacity and reallocation]*.
    ///
    /// If it is important to know the exact allocated capacity of a `Registry`,
    /// always use the [`capacity`] method after construction.
    ///
    /// For `Registry<T>` where `T` is a zero-sized type, there will be no allocation
    /// and the capacity will always be `usize::MAX`.
    ///
    /// [Capacity and reallocation]: #capacity-and-reallocation
    /// [`capacity`]: Registry::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut registry = Registry::with_capacity(10);
    ///
    /// // The registry contains no items, even though it has capacity for more
    /// assert_eq!(registry.count(), 0);
    /// assert!(registry.capacity() >= 10);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     registry.insert(i);
    /// }
    /// assert_eq!(registry.count(), 10);
    /// assert!(registry.capacity() >= 10);
    ///
    /// // ...but this may make the registry reallocate
    /// registry.insert(11);
    /// assert_eq!(registry.count(), 11);
    /// assert!(registry.capacity() >= 11);
    ///
    /// // A registry of a zero-sized type will always over-allocate, since no
    /// // allocation is necessary
    /// let registry_units = Registry::<()>::with_capacity(10);
    /// assert_eq!(registry_units.capacity(), usize::MAX);
    /// ```
    ///
    /// Note, that registries constructed this way will still reallocate on removal,
    /// even if no reallocation would be necessary on insertion.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            indices: Vec::new(),
        }
    }

    /// Returns the total number of elements the registry can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut reg: Registry<i32> = Registry::with_capacity(10);
    /// reg.insert(42);
    /// assert!(reg.capacity() >= 10);
    /// ```
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Inserts an element into the registry and returns its unique index.
    /// As long, as the element is not removed, the index is guaranteeed
    /// to point at the element. Might cause reallocation.
    #[must_use]
    pub fn insert(&mut self, item: T) -> usize {
        if let Some(index) = self.indices.pop() {
            self.data[index] = Some(item);
            index
        } else {
            self.data.push(Some(item));
            self.data.len() - 1
        }
    }

    /// Removes and returns the element at `index`. Guaranteed performance of *O*(1).
    /// Might cause reallocation.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if let Some(value) = self.data.get_mut(index) {
            self.indices.push(index);
            value.take()
        } else {
            None
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e`, for which `f(&e)`
    /// returns `false`.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        for (i, v) in self.data.iter_mut().enumerate() {
            if !v.as_ref().map(&mut f).unwrap_or(false) {
                *v = None;
                self.indices.push(i);
            }
        }
    }

    /// Retains only the elements specified by the predicate,
    /// passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e`, for which `f(&mut e)`
    /// returns `false`.
    pub fn retain_mut<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        for (i, v) in self.data.iter_mut().enumerate() {
            if !v.as_mut().map(|x| f(x)).unwrap_or(false) {
                *v = None;
                self.indices.push(i);
            }
        }
    }

    /// Empties the entire registry. Optimized, never reallocates.
    pub fn clear(&mut self) {
        self.data.clear();
        self.indices.clear();
    }

    /// Returns the count of contained elemetns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len() - self.indices.len()
    }

    /// Returns `true`, if count is 0.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the element at `index`, if such exists.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index).and_then(Option::as_ref)
    }

    /// Returns a mutable reference to the element at `index`, if such exists.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index).and_then(Option::as_mut)
    }

    /// Swaps two elements in the registry.
    ///
    /// # Panics
    /// If indices `a` or `b` are not present in the registry.
    pub fn swap(&mut self, a: usize, b: usize) {
        let x_a = self
            .data
            .get_mut(a)
            .expect("index a is not in the registry")
            .take()
            .expect("index a is not in the registry");

        let x_b = self
            .data
            .get_mut(b)
            .expect("index b is not in the registry")
            .take()
            .expect("index b is not in the registry");

        self.data[a] = Some(x_b);
        self.data[b] = Some(x_a);
    }

    /// Returns `true` if the registry contains an element with the given value.
    #[must_use]
    pub fn contains_value(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.data
            .iter()
            .any(|opt| opt.as_ref().is_some_and(|v| *v == *x))
    }

    /// Returns `true` if the registry contains the given index.
    #[must_use]
    pub fn contains_index(&self, index: usize) -> bool {
        self.data.get(index).is_some()
    }

    /// Creates an iterator over references to the registry's elements.
    /// Of type `(usize, &T)`.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        self.into_iter()
    }

    /// Creates an iterator over mutable references to the registry's elements.
    /// Of type `(usize, &mut T)`.
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.into_iter()
    }

    /// Creates an iterator over the registry's values.
    /// Of type `T`.
    #[must_use]
    pub fn into_values(self) -> IntoValues<T> {
        IntoValues {
            data_iter: self.data.into_iter(),
            free_indices: self.indices.len(),
        }
    }

    /// Creates an iterator over references to the registry's values.
    /// Of type `T`.
    #[must_use]
    pub fn values(&self) -> Values<'_, T> {
        Values {
            data_iter: self.data.iter(),
            free_indices: self.indices.len(),
        }
    }

    /// Creates an iterator over mutable references to the registry's values.
    /// Of type `&mut T`.
    #[must_use]
    pub fn values_mut(&mut self) -> ValuesMut<'_, T> {
        ValuesMut {
            data_iter: self.data.iter_mut(),
            free_indices: self.indices.len(),
        }
    }

    /// Creates an iterator over keys in the registry.
    /// Of type `usize`.
    #[must_use]
    pub fn keys(&self) -> Keys<'_, T> {
        Keys {
            data_iter: self.data.iter().enumerate(),
            free_indices: self.indices.len(),
        }
    }
}

impl<T> Debug for Registry<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> From<[T; N]> for Registry<T> {
    fn from(value: [T; N]) -> Self {
        Self {
            data: value.into_iter().map(Some).collect(),
            indices: Vec::new(),
        }
    }
}

impl<T> FromIterator<T> for Registry<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().map(Some).collect(),
            indices: Vec::new(),
        }
    }
}

impl<T> Index<usize> for Registry<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("no entry found for index")
    }
}

impl<T> IndexMut<usize> for Registry<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("no entry found for index")
    }
}

impl<'a, T> IntoIterator for &'a Registry<T> {
    type Item = (usize, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            data_iter: self.data.iter().enumerate(),
            free_indices: self.indices.len(),
        }
    }
}

impl<'a, T> IntoIterator for &'a mut Registry<T> {
    type Item = (usize, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            data_iter: self.data.iter_mut().enumerate(),
            free_indices: self.indices.len(),
        }
    }
}

impl<T> IntoIterator for Registry<T> {
    type Item = (usize, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            data_iter: self.data.into_iter().enumerate(),
            free_indices: self.indices.len(),
        }
    }
}

impl<T: Serialize> Serialize for Registry<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len()))?;

        for (i, v) in self {
            map.serialize_entry(&i, v)?;
        }

        map.end()
    }
}

struct RegistryVisitor<T>(PhantomData<T>);

impl<T> Default for RegistryVisitor<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for RegistryVisitor<T> {
    type Value = Registry<T>;

    fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
        formatter.write_str("a uint-indexed map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut reg = Registry::with_capacity(map.size_hint().unwrap_or(0));

        while let Some((i, entry)) = map.next_entry()? {
            for j in reg.data.len()..i {
                reg.data.push(None);
                reg.indices.push(j);
            }

            reg.data.push(Some(entry));
        }

        Ok(reg)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Registry<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RegistryVisitor::default())
    }
}

// Iterators

/// An owning iterator over the entries of `Registry<T>`.
/// Item type is `(usize, T)`
#[derive(Debug)]
pub struct IntoIter<T> {
    data_iter: Enumerate<vec::IntoIter<Option<T>>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next_back()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// A reference iterator over the entries of `Registry<T>`.
/// Item type is `(usize, &T)`
#[derive(Debug)]
pub struct Iter<'a, T> {
    data_iter: Enumerate<slice::Iter<'a, Option<T>>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next_back()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// A mutable reference iterator over the entries of `Registry<T>`.
/// Item type is `(usize, &mut T)`
#[derive(Debug)]
pub struct IterMut<'a, T> {
    data_iter: Enumerate<slice::IterMut<'a, Option<T>>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let (index, Some(item)) = self.data_iter.next_back()? {
                return Some((index, item));
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// An owning iterator over the values of `Registry<T>`.
/// Item type is `T`.
#[derive(Debug)]
pub struct IntoValues<T> {
    data_iter: vec::IntoIter<Option<T>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<T> Iterator for IntoValues<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<T> ExactSizeIterator for IntoValues<T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<T> DoubleEndedIterator for IntoValues<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next_back()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// A reference iterator over the values of `Registry<T>`.
/// Item type is `&T`.
#[derive(Debug)]
pub struct Values<'a, T> {
    data_iter: slice::Iter<'a, Option<T>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<'a, T> ExactSizeIterator for Values<'a, T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<'a, T> DoubleEndedIterator for Values<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next_back()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// A mutable reference iterator over the values of `Registry<T>`.
/// Item type is `&mut T`.
#[derive(Debug)]
pub struct ValuesMut<'a, T> {
    data_iter: slice::IterMut<'a, Option<T>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<'a, T> Iterator for ValuesMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<'a, T> ExactSizeIterator for ValuesMut<'a, T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<'a, T> DoubleEndedIterator for ValuesMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.data_iter.next_back()? {
                return Some(item);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}

/// An iterator over the keys of `Registry<T>`.
/// Item type is `usize`.
#[derive(Debug)]
pub struct Keys<'a, T> {
    data_iter: Enumerate<slice::Iter<'a, Option<T>>>,
    /// Remaining free indices.
    free_indices: usize,
}

impl<'a, T> Iterator for Keys<'a, T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (i, Some(_)) = self.data_iter.next()? {
                return Some(i);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl<'a, T> ExactSizeIterator for Keys<'a, T> {
    fn len(&self) -> usize {
        self.data_iter.len() - self.free_indices
    }
}

impl<'a, T> DoubleEndedIterator for Keys<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let (i, Some(_)) = self.data_iter.next_back()? {
                return Some(i);
            }

            // It was a None value.
            self.free_indices -= 1;
        }
    }
}
