/*
 *  Copyright (C) 2026 Team Kodi
 *  This file is part of Kodi - https://kodi.tv
 *
 *  SPDX-License-Identifier: GPL-2.0-or-later
 *  See LICENSES/README.md for more information.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace DVTUNNEL
{

constexpr std::size_t MAX_METADATA_BYTES = 512;
constexpr std::size_t PACKET_BYTES = 128;
constexpr std::size_t MAX_PACKETS = 5;

/*!
 * \brief Source colour parameters in the fixed-point form of the transport
 * header.
 */
struct DvColourParams
{
  std::int32_t ycc[9] = {};
  std::uint32_t offset[3] = {};
  std::int32_t lms[9] = {};
  std::uint8_t colorSpace = 0;
};

/*!
 * \brief Canvas rectangle, right and bottom exclusive.
 */
struct DvRect
{
  unsigned left = 0;
  unsigned top = 0;
  unsigned right = 0;
  unsigned bottom = 0;

  bool IsEmpty() const { return left >= right || top >= bottom; }
};

/*!
 * \brief Serialized display management metadata and the colour parameters it
 * carries.
 */
struct DvDisplayMetadata
{
  std::vector<std::uint8_t> bytes;
  DvColourParams colour;
};

using DvPacket = std::array<std::uint8_t, PACKET_BYTES>;

/*!
 * \brief Serializes the display management metadata of a picture.
 *
 * Level 6 blocks are dropped; the other extension blocks are rebuilt from
 * FFmpeg's parsed fields.
 *
 * \param blob an AVDOVIMetadata side-data blob
 * \param blobSize size of blob in bytes
 * \return the metadata, or empty if the blob is unreadable or a value does not
 *         fit the transport
 */
std::optional<DvDisplayMetadata> SerializeDisplayMetadata(const void* blob, std::size_t blobSize);

/*!
 * \brief Where a picture sits on the carrier canvas.
 */
struct DvPlacement
{
  unsigned sourceWidth = 0;
  unsigned sourceHeight = 0;
  unsigned canvasWidth = 0;
  unsigned canvasHeight = 0;
  DvRect picture;
};

/*!
 * \brief Sets the level 5 active area of serialized metadata.
 *
 * The level 5 offsets, in source pixels, are moved to the picture's place on
 * the canvas; metadata without a level 5 block gets one for the canvas outside
 * the picture. Without a placement the whole canvas is active.
 *
 * \return false if the block walk is inconsistent or the picture is not on the
 *         canvas
 */
bool SetLevel5ActiveArea(std::vector<std::uint8_t>& bytes, const DvPlacement* placement);

/*!
 * \brief Number of packets that carry metadataBytes of metadata.
 */
std::size_t PacketCount(std::size_t metadataBytes);

/*!
 * \brief Splits serialized metadata into CRC-protected transport packets.
 *
 * \param metadataId 0..15, changed by the caller whenever the metadata changes
 * \return the packets, or empty if metadata is empty or longer than
 *         MAX_METADATA_BYTES
 */
std::optional<std::vector<DvPacket>> Packetize(const std::vector<std::uint8_t>& metadata,
                                               unsigned metadataId = 0);

/*!
 * \brief CRC-32/MPEG-2 of data.
 */
std::uint32_t Crc32Mpeg2(const std::uint8_t* data, std::size_t size);

} // namespace DVTUNNEL
