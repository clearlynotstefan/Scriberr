package transcription

import (
	"math"
	"sort"
	"strings"

	"scriberr/internal/transcription/interfaces"
)

// mergeExclusiveDiarization uses Community-1's non-overlapping speaker turns
// instead of flattening an entire coarse ASR chunk to its majority speaker.
// When the ASR backend has no word timestamps (Canary Transformers), word
// positions are explicitly marked as estimates.
func (u *UnifiedTranscriptionService) mergeExclusiveDiarization(transcript *interfaces.TranscriptResult, diarization *interfaces.DiarizationResult) *interfaces.TranscriptResult {
	merged := *transcript
	merged.Metadata = cloneStringMap(transcript.Metadata)
	merged.Metadata["diarization_output"] = "exclusive"

	if len(transcript.WordSegments) > 0 {
		merged.WordSegments = append([]interfaces.TranscriptWord(nil), transcript.WordSegments...)
		for i := range merged.WordSegments {
			speaker := u.findBestSpeakerForSegment(merged.WordSegments[i].Start, merged.WordSegments[i].End, diarization.Segments)
			if speaker != "" {
				merged.WordSegments[i].Speaker = stringPointer(speaker)
			}
		}
		merged.Segments = groupWordsBySpeaker(merged.WordSegments)
		for _, segment := range transcript.Segments {
			if strings.TrimSpace(segment.Text) == "" {
				merged.Segments = append(merged.Segments, segment)
			}
		}
		sortTranscriptSegments(merged.Segments)
		return &merged
	}

	merged.Metadata["timestamp_precision"] = "speaker_turn_estimate"
	merged.Metadata["word_timestamps"] = "estimated_from_exclusive_diarization"
	merged.Segments = nil
	merged.WordSegments = nil

	for _, segment := range transcript.Segments {
		if strings.TrimSpace(segment.Text) == "" {
			// Empty chunks are meaningful evidence that the ASR backend processed
			// this span. Keep them instead of silently closing the timeline gap.
			merged.Segments = append(merged.Segments, segment)
			continue
		}

		turns := clippedSpeakerTurns(segment.Start, segment.End, diarization.Segments)
		words := strings.Fields(segment.Text)
		if len(turns) == 0 || len(words) == 0 {
			merged.Segments = append(merged.Segments, segment)
			continue
		}

		allocations := allocateWordsByDuration(len(words), turns)
		wordIndex := 0
		for turnIndex, turn := range turns {
			count := allocations[turnIndex]
			if count == 0 {
				continue
			}

			turnWords := words[wordIndex : wordIndex+count]
			wordIndex += count
			speaker := turn.Speaker
			merged.Segments = append(merged.Segments, interfaces.TranscriptSegment{
				Start:    turn.Start,
				End:      turn.End,
				Text:     strings.Join(turnWords, " "),
				Speaker:  &speaker,
				Language: segment.Language,
			})

			duration := turn.End - turn.Start
			for wordOffset, word := range turnWords {
				start := turn.Start + duration*float64(wordOffset)/float64(count)
				end := turn.Start + duration*float64(wordOffset+1)/float64(count)
				merged.WordSegments = append(merged.WordSegments, interfaces.TranscriptWord{
					Start: start, End: end, Word: word, Score: 0, Speaker: &speaker,
				})
			}
		}
	}

	return &merged
}

func clippedSpeakerTurns(start, end float64, segments []interfaces.DiarizationSegment) []interfaces.DiarizationSegment {
	turns := make([]interfaces.DiarizationSegment, 0)
	for _, segment := range segments {
		clippedStart := max(start, segment.Start)
		clippedEnd := min(end, segment.End)
		if clippedEnd <= clippedStart {
			continue
		}
		segment.Start = clippedStart
		segment.End = clippedEnd
		turns = append(turns, segment)
	}
	sort.SliceStable(turns, func(i, j int) bool { return turns[i].Start < turns[j].Start })
	return turns
}

func allocateWordsByDuration(wordCount int, turns []interfaces.DiarizationSegment) []int {
	allocations := make([]int, len(turns))
	if wordCount == 0 || len(turns) == 0 {
		return allocations
	}

	totalDuration := 0.0
	for _, turn := range turns {
		totalDuration += turn.End - turn.Start
	}
	if totalDuration <= 0 {
		allocations[len(allocations)-1] = wordCount
		return allocations
	}

	allocated := 0
	cumulativeDuration := 0.0
	for i, turn := range turns {
		if i == len(turns)-1 {
			allocations[i] = wordCount - allocated
			break
		}
		cumulativeDuration += turn.End - turn.Start
		target := int(math.Round(float64(wordCount) * cumulativeDuration / totalDuration))
		if wordCount >= len(turns) {
			target = maxInt(target, allocated+1)
			target = minInt(target, wordCount-(len(turns)-i-1))
		}
		allocations[i] = maxInt(0, target-allocated)
		allocated += allocations[i]
	}
	return allocations
}

func groupWordsBySpeaker(words []interfaces.TranscriptWord) []interfaces.TranscriptSegment {
	if len(words) == 0 {
		return nil
	}

	segments := make([]interfaces.TranscriptSegment, 0)
	start := 0
	for i := 1; i <= len(words); i++ {
		if i < len(words) && speakerValue(words[i].Speaker) == speakerValue(words[start].Speaker) {
			continue
		}
		group := words[start:i]
		segments = append(segments, interfaces.TranscriptSegment{
			Start:   group[0].Start,
			End:     group[len(group)-1].End,
			Text:    joinTranscriptWords(group),
			Speaker: group[0].Speaker,
		})
		start = i
	}
	return segments
}

func joinTranscriptWords(words []interfaces.TranscriptWord) string {
	parts := make([]string, 0, len(words))
	for _, word := range words {
		if value := strings.TrimSpace(word.Word); value != "" {
			parts = append(parts, value)
		}
	}
	return strings.Join(parts, " ")
}

func sortTranscriptSegments(segments []interfaces.TranscriptSegment) {
	sort.SliceStable(segments, func(i, j int) bool { return segments[i].Start < segments[j].Start })
}

func cloneStringMap(source map[string]string) map[string]string {
	clone := make(map[string]string, len(source)+3)
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func stringPointer(value string) *string { return &value }

func speakerValue(speaker *string) string {
	if speaker == nil {
		return ""
	}
	return *speaker
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
