import unittest
from urllib.parse import urlparse

from src.lib.bktree import BKTree, levenshtein_distance
from src.lib.email import domains_from_urls
from src.lib.email_address import parse_email_address
from src.lib.feature_extract import (
    capital_words_ratio,
    count_ip_addresses,
    count_typosquatted_domains,
    count_whitelisted_addresses,
    email_domain_matches_url,
    find_suspicious_words,
    is_ip_address,
    is_typosquatted_domain,
    money_tokens_ratio,
    score_suspicious_words,
)


def test_kernel(x: float) -> float:
    """Returns `2 - x^2`."""
    return 2 - (x * x)


class TestFeatureExtract(unittest.TestCase):
    def test_count_whitelisted_addresses(self):
        count = count_whitelisted_addresses(
            [
                parse_email_address("safe@example.com"),
                parse_email_address("scam@unsafe.com"),
                parse_email_address("gov@trusted.org"),
                parse_email_address("gov@trusted.org.scam"),
                parse_email_address("gov@fbi.trusted.org"),
            ],
            {"example.com", "trusted.org"},
        )
        self.assertEqual(count, 3)

    def test_find_suspicious_words(self):
        actual = list(find_suspicious_words(["hi", "HoW", "are", "yOU"], {"hi", "you"}))
        expected = [0, 3]
        self.assertListEqual(actual, expected)

        actual = list(find_suspicious_words(["hi", "HoW", "are", "yOU"], {"free"}))
        expected = []
        self.assertListEqual(actual, expected)

        actual = list(find_suspicious_words([], {"hi", "you"}))
        expected = []
        self.assertListEqual(actual, expected)

        actual = list(find_suspicious_words(["hi", "HoW", "are", "yOU"], set()))
        expected = []
        self.assertListEqual(actual, expected)

    def test_score_suspicious_words(self):
        self.assertAlmostEqual(
            score_suspicious_words(["hi"], {"hi", "you"}, test_kernel),
            2.0,
        )
        self.assertAlmostEqual(
            score_suspicious_words(["hi", "HoW", "are", "yOU"], set(), test_kernel),
            0.0,
        )
        self.assertAlmostEqual(
            score_suspicious_words([], {"hi"}, test_kernel),
            0.0,
        )
        self.assertAlmostEqual(
            score_suspicious_words(
                ["hi", "HoW", "are", "yOU"],
                {"hi", "you"},
                test_kernel,
            ),
            0.75,
        )

        # This function should be invariant under input size
        self.assertAlmostEqual(
            score_suspicious_words(
                (["hi"] * 1000) + (["bye"] * 1000),
                {"hi"},
                test_kernel,
            ),
            score_suspicious_words(
                (["hi"] * 10_000) + (["bye"] * 10_000),
                {"hi"},
                test_kernel,
            ),
            places=3,
        )

    def test_is_typosquatted_domain(self):
        tree = BKTree(levenshtein_distance, ["example.com", "test.com", "sample.org"])

        self.assertTrue(is_typosquatted_domain("examble.com", tree, 1))
        self.assertTrue(is_typosquatted_domain("test.co", tree, 1))
        self.assertTrue(is_typosquatted_domain("snple.org", tree, 2))

        self.assertFalse(is_typosquatted_domain("snpe.org", tree, 2))
        self.assertFalse(is_typosquatted_domain("facebook.com", tree, 3))
        self.assertFalse(is_typosquatted_domain("example.com", tree, 3))

        self.assertEqual(
            count_typosquatted_domains(
                domains_from_urls(
                    {
                        urlparse("http://abcd.org"),
                        urlparse("http://example.com.net"),
                        urlparse("http://www.smple.org"),
                        urlparse("https://example.com"),
                        urlparse("https://ex4mple.com"),
                    }
                ),
                tree,
                1,
            ),
            2,
        )
        self.assertEqual(count_typosquatted_domains([], tree, 1), 0)

        tree = BKTree(levenshtein_distance, [])
        self.assertFalse(is_typosquatted_domain("anything.com", tree, 100))
        self.assertFalse(is_typosquatted_domain("", tree, 1))

    def test_is_ip_address(self):
        self.assertTrue(is_ip_address(urlparse("http://1.1.1.1")))
        self.assertTrue(is_ip_address(urlparse("http://1.2.3.4/abc/e?q=1#frag")))
        self.assertTrue(is_ip_address(urlparse("http://[::]/abc/e?q=1#frag")))
        self.assertTrue(is_ip_address(urlparse("https://[::1]/abc/e?q=1#frag")))

        self.assertFalse(is_ip_address(urlparse("")))
        self.assertFalse(is_ip_address(urlparse("https://c.d.uk.edu")))
        self.assertFalse(is_ip_address(urlparse("https://a.b.com/abc/e?q=1#frag")))
        self.assertFalse(is_ip_address(urlparse("https://1.1.1.1.edu")))
        self.assertFalse(is_ip_address(urlparse("https://test.com/1.1.1.1")))

        self.assertEqual(
            count_ip_addresses(
                [
                    urlparse("http://1.1.1.1"),
                    urlparse("http://1.2.3.4/abc/e?q=1#frag"),
                    urlparse(""),
                    urlparse("https://1.1.1.1.edu"),
                ]
            ),
            2,
        )
        self.assertEqual(count_ip_addresses([]), 0)

    def test_email_domain_matches_url(self):
        self.assertTrue(
            email_domain_matches_url(
                parse_email_address("safe@example.com"),
                domains_from_urls(
                    {urlparse("http://abcd.org"), urlparse("http://net.example.com")}
                ),
            )
        )
        self.assertTrue(
            email_domain_matches_url(
                parse_email_address("safe@1.1.1.1"),
                domains_from_urls({urlparse("http://1.1.1.1")}),
            )
        )
        self.assertTrue(
            email_domain_matches_url(parse_email_address("safe@example.com"), [])
        )

        self.assertFalse(
            email_domain_matches_url(
                parse_email_address("safe@example.com"),
                domains_from_urls({urlparse("https://abcd.org")}),
            )
        )
        self.assertFalse(
            email_domain_matches_url(
                parse_email_address("safe@example.com"),
                domains_from_urls(
                    {urlparse("http://abcd.org"), urlparse("http://example.com.net")}
                ),
            )
        )
        self.assertFalse(email_domain_matches_url(None, []))

    def test_capital_words_ratio(self):
        # We expect the values to come from the exact fractions below,
        # so we can use assertEqual instead of assertAlmostEqual for stronger tests.
        self.assertEqual(capital_words_ratio([]), 0 / 1)
        self.assertEqual(capital_words_ratio(["hello", "world"]), 0 / 2)
        self.assertEqual(capital_words_ratio(["Hello", "world"]), 0 / 2)
        self.assertEqual(capital_words_ratio(["HELLO", "world"]), 1 / 2)
        self.assertEqual(capital_words_ratio(["HELLO", "WORLD"]), 2 / 2)
        self.assertEqual(capital_words_ratio(["HELLO", "foO", "WORLD"]), 2 / 3)
        self.assertEqual(capital_words_ratio(["H3LL0", "FoO", "WORLD1"]), 0 / 3)
        self.assertEqual(capital_words_ratio(["0", "1", "WORLD2"]), 0 / 3)
        self.assertEqual(capital_words_ratio(["0", "", "WORLD"]), 1 / 3)

    def test_money_tokens_ratio(self):
        # We expect the values to come from the exact fractions below,
        # so we can use assertEqual instead of assertAlmostEqual for stronger tests.
        self.assertEqual(money_tokens_ratio([]), 0 / 1)
        self.assertEqual(money_tokens_ratio(["hey", "$"]), 0 / 2)
        self.assertEqual(money_tokens_ratio(["hey", "f$€£"]), 0 / 2)
        self.assertEqual(money_tokens_ratio(["free", "$100"]), 1 / 2)
        self.assertEqual(money_tokens_ratio(["free", "$100.69"]), 1 / 2)
        self.assertEqual(money_tokens_ratio(["free", "$100aaaaa"]), 1 / 2)
        self.assertEqual(money_tokens_ratio(["free", ""]), 0 / 2)
        self.assertEqual(
            money_tokens_ratio(["free", "$100", "or", "€1M!", "jackpot"]), 2 / 5
        )
